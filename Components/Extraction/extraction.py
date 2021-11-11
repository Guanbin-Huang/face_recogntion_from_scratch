import cv2
import torch
import copy




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
        img = letterbox(img0, new_shape = imgsz[0]) #todo letterbox

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
        pre = non_max_suppression_face(pre, conf_thres, iou_thres)

        # 5. return bboxes across given images in a list -------------------------------------
        for i, dets in enumerate(pred): # detections per image
            gn = torch.tensor(raw_img.shape)[[1,0,1,0]].to(self.device) # gain of bboxes for normalization      whwh
            gn_lmks = torch.tensor(raw_img.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device) # gain of landmarks for normalization  xy

            if len(dets): # if the current image has det
                # rescale boxes from img_size to img0 size ---------------------------------------------------------------------------------------
                dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], raw_img.shape).round() # img1_shape, coord, img0_shape #todo scale_coords

                # print results
                for cls in dets[:, -1].unique():
                    n = (dets[:, -1] == c).sum() # detections per class

                dets[:, 5: 15] = scale_coords_landmarks(img.shape[2:], dets[:, 5:15], raw_img.shape).round() # round --> rounding off
                # guess that det [cx,cy,w,h,  conf , 5 landmarks x 2(xy), label]

                bboxes_list = []

                # get the info of the bboxes and visualize --------------------------------------------------------------------------------------
                for i_det in range(dets.size()[0]):
                    xywh = xyxy2xywh(dets[j, :4].view(1, 4) / gn).view(-1).tolist() # view(-1) can be seen as missing a 1, just flattern layer.  ref: https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view#:~:text=return%20input.view(input.size(0)%2C%20-1)
                    conf = dets[i_det, 4]
                    landmarks = (dets[i_det, 5:15].view(1, 10) / gn_lmks).view(-1).tolist()
                    class_idx = dets[i_det, -1].cpu().numpy()

                    a_bbox_info = pack_up_bbox_result(raw_img, xywh, conf, landmarks, i_det) # by offerering some info of the bbox, pack up the box on the image and visualiaze(optional) #todo show_results

                    bboxes_list.append(a_bbox_info)

            else:
                bboxes_list = None

        return bboxes_list






        # endregion NMS









