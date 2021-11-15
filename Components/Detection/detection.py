# only for unit test debug
import os

import cv2
import torch
import sys

from yolov5_face_master.utils.datasets import letterbox
from yolov5_face_master.utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from yolov5_face_master.utils.torch_utils import time_synchronized


def load_model(weight_path, device):
    model = torch.load(weight_path, map_location=device)["model"].float().fuse().eval() # load FP32 model rather than float64
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pads = None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    
    return coords #todo we gonna use warpaffine

def pack_up_bbox_result(img, xywh, conf, landmarks, index):
    # info of the bbox
    h, w, c = img.shape
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    
    a_bbox_info = []
    
    # settings
    thickness_line = 1 or round(0.002 * (h + w) / 2) + 1  # font thickness
    thickness_font = max(thickness_line - 1, 1)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    # get_xyrb and draw rect
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2), int(y2)), (0,255,0), thickness=thickness_line, lineType=cv2.LINE_AA)
    a_bbox_info.append(x1)
    a_bbox_info.append(y1)
    a_bbox_info.append(x2)
    a_bbox_info.append(y2)


    # get_landmarks and draw them
    for i in range(5):
        x_lmk = int(landmarks[2 * i] * w)
        y_lmk = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (x_lmk, y_lmk), thickness_line+1, colors[i], -1)
        a_bbox_info.append(x_lmk)
        a_bbox_info.append(y_lmk)

    label = str(conf)[:5]

    cv2.putText(img, label, (x1, y1 - 2), 0, thickness_line / 3, [225, 255, 255], thickness=thickness_font, lineType=cv2.LINE_AA)
    
    return a_bbox_info # (x,y,r,b, 5 x_lmks, 5 y_lmks)


class Detector:
    def __init__(self, net_param):
        """
        :param net_param: a path   e.g yolov5/best.pt
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # at first only consider single GPU
        self.model  = load_model(net_param, self.device)
        self.model.eval() # always remebmer to do it when doing inference

    def __call__(self, raw_img, *args, **kwargs):
        # 1. predefined var ---------------------------------------------------
        img_size = 800
        conf_thres = 0.3
        iou_thres = 0.5  # hard code at fisrt, which can be changed in the future.

        # region 2. preprocessing ---------------------------------------------
        # 2.0 resize -----------------------------------------------------------
        img0 = raw_img.copy()

        h0, w0 = img0.shape[:2]
        ratio = img_size / max(h0, w0)

        if ratio != 1: # always resize down, only resize up if training with aug
            interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR # resize down using inter_area   resize up using linear
            img0 = cv2.resize(img0, (int(w0 * ratio), int(h0 * ratio)), interpolation=interp)  # let the maxline of the img0 be img_size

        # 2.1 exact division by max stride
        imgsz = check_img_size(img_size, s = self.model.stride.max()) #todo check_img_size

        # 2.2 letterbox
        img = letterbox(img0, new_shape = imgsz)[0] #todo letterbox

        # 2.3 common 4 steps: channel permute, to tensor, to float, normalize, unsqueeze
        img = img[..., ::-1].transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 225.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # endregion preprocessing

        # 3. inference -------------------------------------------------------------
        t1 = time_synchronized()
        pred = self.model(img)[0]

        # 4. NMS ---------------------------------------------------------------
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # 5. return bboxes across given images in a list -------------------------------------
        for i, dets in enumerate(pred): # detections per image
            gn = torch.tensor(raw_img.shape)[[1,0,1,0]].to(self.device) # gain of bboxes for normalization      whwh
            gn_lmks = torch.tensor(raw_img.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device) # gain of landmarks for normalization  xy

            if len(dets): # if the current image has det
                # rescale boxes from img_size to img0 size ---------------------------------------------------------------------------------------
                dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], raw_img.shape).round() # img1_shape, coord, img0_shape #todo scale_coords

                # print results
                for cls in dets[:, -1].unique():
                    n = (dets[:, -1] == cls).sum() # detections per class

                dets[:, 5: 15] = scale_coords_landmarks(img.shape[2:], dets[:, 5:15], raw_img.shape).round() # round --> rounding off
                # guess that det [cx,cy,w,h,  conf , 5 landmarks x 2(xy), label]

                bboxes_list = []

                # get the info of the bboxes and visualize --------------------------------------------------------------------------------------
                for i_det in range(dets.size()[0]):
                    xywh = xyxy2xywh(dets[i_det, :4].view(1, 4) / gn).view(-1).tolist() # view(-1) can be seen as missing a 1, just flattern layer.  ref: https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view#:~:text=return%20input.view(input.size(0)%2C%20-1)
                    conf = dets[i_det, 4].cpu().numpy()
                    landmarks = (dets[i_det, 5:15].view(1, 10) / gn_lmks).view(-1).tolist()
                    class_num = dets[i_det, 15].cpu().numpy()
                    a_bbox_info = pack_up_bbox_result(raw_img, xywh, conf, landmarks, i_det) # by offerering some info of the bbox, pack up the box on the image and visualiaze(optional) #todo show_results

                    bboxes_list.append(a_bbox_info)

            else:
                bboxes_list = None

        return bboxes_list


if __name__ == "__main__":
    # unit test in win
    os.chdir(r"D:\huang_face_det")
    sys.path.append(r"./Components/Detection/yolov5_face_master") #! a disgusting bug torch.load should share a same path as torch.save. Or put them in the sys.path together.

    # start
    # init detector and input image
    ckpt_path = r"C:\Users\PeterHuang\Desktop\raw_st_FaceRecognizition\yolov5_face_master\runs\train\exp5\weights\best.pt"
    detector = Detector(ckpt_path)
    
    img_path = r"./imgs_edu/yy_czy.jpg"
    img = cv2.imread(img_path)

    bboxes_list = detector(img)
    a = 1














