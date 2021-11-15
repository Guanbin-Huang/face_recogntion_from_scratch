# -*- coding: utf-8 -*-
'''
1、输入一张图片，读取后，crop为128*128，输入net，输出1*1024特征向量，与人脸库做cos，根据阈值判断是否相似。
'''

from __future__ import print_function
import os
import cv2
from arcface_pytorch.models import *
import torch
import numpy as np
import time
from arcface_pytorch.config import Config
from torch.nn import DataParallel
from yolov5_face_master import detect_arc_face
from PIL import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont

# import sys
# sys.path.append(r"D:\pycharmproject\FaceRecognizition\yolov5_face_master")

def get_featurs(model, crop_img):
    
    # for i, img_path in enumerate(test_list):
        # load_image函数有点不一样：
        #   合并image本身+image的左右翻转图片为2张图片
        # 因此，在这里的得到的image.shape=(2, 1, 128, 128)
    image = load_image(crop_img)
    if image is None:
        print('read {} error'.format(crop_img))
    

    data = torch.from_numpy(image)
    data = data.to(torch.device("cuda"))
    output = model(data)
    output = output.data.cpu().numpy()

    # fe_1为image本身的512维特征，fe_2为image的左右翻转图片的512维特征
    # 对于每张图片，合并本身512维特征+左右翻转的512维特征，得到一个1024维的特征作为该图片的feature
    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))
    # print(feature.shape)

    return feature

def load_image(img):
    # image = cv2.imread(img, 0)
    image = img.convert("L")
    image = np.array(image)
    h, w = image.shape
    x_min = min(h, w)
    image = image[h-x_min: h, 0: w]
    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    if image is None:
        return None
    # image.shape=(128, 128)
    # 合并image+image的左右翻转图片
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    # 最终返回的image.shape=(2, 1, 128, 128)
    return image


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    return cosa

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # 判断是否为opencv图片类型
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(r"C:\Windows\Fonts\seguisb.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText, spacing=8)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':

    # 1.创建模型
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()


    # 2.加载模型参数
    model = DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_model(model, opt.test_model_path)
    # model.load_state_dict(torch.load(opt.test_model_path))
    model.load_state_dict(torch.load(opt.test_model_path, map_location=device))
    model.to(device)

    model.eval()

    detect = detect_arc_face.load_model(r"D:\pycharmproject\FaceRecognizition\yolov5_face_master\runs\train\exp5\weights\best.pt", device)

    # feature_org = torch.load(r"D:\pycharmproject\FaceRecognizition\arcface_pytorch\fet_data")
    detect_arc_face.detect_one(detect, device, model)

    # exit()
    # bboxes = bbox_image[0]
    # image = bbox_image[1]
    # for box in bboxes:
    #     x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    #
    # # img = img.squeeze()
    # # img_np = img.cpu().numpy()
    # # crop_img = img.copy()[..., ::-1]
    #
    #     crop_img = Image.fromarray(image).crop((x1, y1, x2, y2))
    #
    #     feature_test = get_featurs(model, crop_img)
    #
    #     cls_list = []
    #     max_score = []
    #     for cls in feature_org:
    #         # print(cls)
    #         cls_score = []
    #         # MTCNN每识别一个框，cls_list中四个类别
    #         cls_list.append(cls)
    #         for fet in feature_org[cls]:
    #             # 将框和存储的每个人的多个特征一一对比
    #             score = compare(torch.tensor(feature_test), torch.tensor(fet))
    #             print(score)
    #             cls_score.append(score)
    #         # print(cls_score)
    #         # 每个存储类别取出网络认为每个框中最大的余弦相似度值
    #         max_score.append(max(cls_score))
    #
    #     if max(max_score) > 0.3:
    #         print(max(max_score))
    #         index = np.argmax(max_score)
    #         # print(index)
    #         pred_cls = cls_list[index]
    #
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         cv2.putText(image, pred_cls, (x1, y1 - 2), 0, 3, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)

