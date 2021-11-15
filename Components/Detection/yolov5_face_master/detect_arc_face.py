# -*- coding: UTF-8 -*-
import pickle
import time
import cv2
import torch
import numpy as np

import copy

from PIL import Image
from yolov5_face_master.models.experimental import attempt_load
from yolov5_face_master.utils.datasets import letterbox
from yolov5_face_master.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5_face_master.utils.plots import plot_one_box
from yolov5_face_master.utils.torch_utils import select_device, load_classifier, time_synchronized

from detector import get_featurs, load_image, compare
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, index):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    # cv2.imwrite("image.jpg", img)
# 此处加的是为了第一步的人脸crop
#     from PIL import Image
#
#     crop_img = img.copy()[...,::-1]
#     crop_img = Image.fromarray(crop_img).crop((x1, y1, x2, y2))
#     save_index = random.randint(1, 1000)
#
#     crop_img.save(fr"D:\pycharmproject\FaceRecognizition\arcface_pytorch\face_data\hgb\{save_index}.jpg")


    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    bbox_list = []
    bbox_list.append(x1)
    bbox_list.append(y1)
    bbox_list.append(x2)
    bbox_list.append(y2)
    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)
        bbox_list.append(point_x)
        bbox_list.append(point_y)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return bbox_list




def detect_one(model, device, arc_model):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    '''
        cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，
        如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        '''
    while True:
        t0 = time.time()
        ret, frame = cap.read()

        img0 = copy.deepcopy(frame)

        h0, w0 = img0.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        print('img.shape: ', img.shape)
        print('orgimg.shape: ', frame.shape)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                bboxes = []
                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    bbox_list = show_results(frame, xywh, conf, landmarks, j)
                    bboxes.append(bbox_list)

        feature_org = torch.load(r"D:\pycharmproject\FaceRecognizition\arcface_pytorch\feature_list")



        for box in bboxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # img = img.squeeze()
            # img_np = img.cpu().numpy()
            # crop_img = img.copy()[..., ::-1]

            crop_img = Image.fromarray(frame).crop((x1, y1, x2, y2))

            feature_test = get_featurs(arc_model, crop_img)

            cls_list = []
            max_score = []
            for cls in feature_org:
                # print(cls)
                cls_score = []
                # MTCNN每识别一个框，cls_list中四个类别
                cls_list.append(cls)
                for fet in feature_org[cls]:
                    # 将框和存储的每个人的多个特征一一对比
                    score = compare(torch.tensor(feature_test), torch.tensor(fet))
                    print(score)
                    cls_score.append(score)
                # print(cls_score)
                # 每个存储类别取出网络认为每个框中最大的余弦相似度值
                max_score.append(max(cls_score))

            if max(max_score) > 0.2:
                print(max(max_score))
                index = np.argmax(max_score)
                # print(index)
                pred_cls = cls_list[index]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, pred_cls, (x1, y1 - 2), 0, 3, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)

        cv2.imshow("1", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        t1 = time.time()
        print("count------>",t1 - t0)
    cap.release()
    cv2.destroyAllWindows()




