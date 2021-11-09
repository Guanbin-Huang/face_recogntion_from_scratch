import time
import cv2
import torch
import json

import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import sys
import os


class FaceRecognizer:
        def __init__(self, face_det_weight, feature_extract_weight, save_crop_img_dir, feature_dict, use_milvus):
            super(FaceRecognizer, self).__init__()

            # initialize the face detector
            self.detector = Detector.YoLov5(face_det_weight)

            # initialize the face feature extractor
            self.extractor = Extractor.ArcFace(feature_extract_weight)

            # the dir to save the face
            self.save_img_dir = save_crop_img_dir
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # the dir to save the feature vector
            self.feature_dict_path = feature_dict
            self.json_read    = json.load(open(self.feature_dict_path, "r"))


        def Register(self, souce, user_name):
            ...

        def ImageRecognize(self, source):
            ...

        def VideoRecognize(self, source):
            ...

        def align(self, M, image):
            ...


if __name__ == "main":
    # ------------------------------------------- arguments ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(r"--face_det_weight", type = str, default =
                        r"yolov5_face_master/best.pt")
    parser.add_argument(r"--feature_dict", type = str, default = r"arcface_pytorch\id_name.json"
                        , help= r"save several feature for each person")
    parser.add_argument(r"--save_crop_dir", type = str, default = r"arcface_pytorch\face_data"
                        , help= r"save some pictures for a given person")
    parser.add_argument("--source", type = str, defulat = 0, help = r"source of the input, video, image, rtsp, webcam etc")
    parser.add_argument("--use_milvus", action="store_true", help="database use milvus")
    parser.add_argument("--milvus", action="store_true", help="milvus open")
    parser.add_argument("--user_name", type=str, help="the name of the person regestering")

    args = parser.parse_args()
    # ------------------------------------------- arguments ------------------------------------------------

    # start up the face recognition
    face = FaceRecognizer(args.face_det_weight, args.save_crop_dir, args.feature_dict, args.use_milvus)

    # register faces(optional)
    face.Register(args.url, args.user_name)

    # ------------------------------------------- basic tasks ----------------------------------------------
    # from image
    face.ImageRecognize(args.source)

    # from video
    face.VideoRecognize(args.source)

