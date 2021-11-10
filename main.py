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

"""
Notes: 
    We already design the whole framework. It's time to finish each block.
    The easiest block might be get_alignment_matrix.
    
    ref:
        
"""



def compare(face1, face2): #todo  comments can be added.
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    cos_similarity = torch.matmul(face1_norm, face2_norm)
    return cos_similarity



class FaceRecognizer:
        def __init__(self, face_det_weight, feature_extract_weight, save_crop_img_dir, feature_dict, use_milvus):
            super(FaceRecognizer, self).__init__()

            # initialize the face detector
            self.detector = Detector.YoLov5(face_det_weight)

            # initialize the face feature extractor
            self.extractor = Extractor.ArcFace(feature_extract_weight)

            # the dir to save the face
            self.save_std_cop_img_dir = save_crop_img_dir
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # the dir to save the feature vector
            self.feature_dict_path = feature_dict
            self.feat_dict    = json.load(open(self.feature_dict_path, "r"))

        #region ---------------------------------------  some utils -------------------------------------
        def img_to_milvus_feature(self):
            ...

        def img_to_json_feature(self):
            ...

        def crop_align_img_from_vid(self, source):            # used in Register
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#:~:text=%E2%97%86%C2%A0-,VideoCaptureAPIs,-enum%20cv%3A%3AVideoCaptureAPIs

            counter = 0 # count frame
            std_crop_img_list = []

            while True: # for loop the frames
                ret, frame = cap.read()
                if not ret:
                    continue
                counter += 1
                bboxes = self.detector(frame.copy()) # detector contains drawing. So we use frame.copy to avoid the modification of the raw_img

                for box in bboxes:
                    if counter % 6 == 0: # crop a frame every 6 frames
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                        M = self.get_alignment_matrix(box)
                        aligned_img = self.align(M, frame.copy())

                        std_crop_img = aligned_img[...,::-1] # we need to imshow, so reverse the channels.
                        std_crop_img = Image.fromarray(std_crop_img)
                        std_crop_img_list.append(std_crop_img)

                cv2.putText(frame, "Registering, persons except the user keep away", (10, 10), 0, 5, [0,255,0], thickness = 3, lineType = cv2.LINE_AA)
                cv2.imshow("1", frame) # no need to show the aligned image because now it's registering time


            cap.release()
            cv2.destroyAllWindows()
            return std_crop_img_list



        def save_aligned_img_from_vid(self, std_crop_img_list, name):
            """
            save std_crop_img to check for a better debug experience
            """
            # create dir for each user name
            name_dir = os.path.join(self.save_std_cop_img_dir, name) # e.g. /face_recog/.../face_data/ + hgb/

            if not os.path.exists(name_dir) # if hgb dir doesn't exist
                os.mkdir(name_dir)

            # save the std_crop_img in it
            for i, img in enumerate(std_crop_img_list):
                img_path = os.path.join(name_dir, f"{i+1}.jpg")
                img.save() # PIL image.save api # ref: https://omz-software.com/pythonista/docs/ios/Image.html#:~:text=The%20Image%20Module,-The%20Image%20module



        #endregion ------------------------------------- some utils -------------------------------------

        #region -------------------------------------- 3 Major Functionalities ------------------------------------------

        def Register(self, source, user_name):
            # get the input image from the croped image list
            crop_img_list = self.crop_align_img_from_vid(source)   # assume the func is really smart such that it detects the and crop the face

            # save the image to a specific path
            self.save_aligned_img_from_vid(crop_img_list, user_name) # assume crop_img_list contains several PIL image.

            if self.use_milvus:
                #self.img_to_milvus_feature
                ...  # todo
            else:
                # get n x 128 features given n x images
                feature_list = self.img_to_json_feature(crop_img_list)

                # save the feature with its user name
                self.feat_dict[user_name] = feature_list

                # save then as json file
                json.dump(self.feat_dict, open(self.feature_dict_path, "w")) # dump the read_json into a new place(aka. self.feature_dict_path)

        def ImageRecognize(self, source):
            image = cv2.imread(source)
            bboxes = self.detector(image) # 1 images ---> n faces

            if bboxes:
                for box in bboxes:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    M = self.get_alignment_matrix(box)
                    aligned_croped_img = self.crop_and_align(M, image.copy())
                    aligned_croped_img = Image.fromarray(aligned_croped_img)
                    feature_test = self.extractor(aligned_croped_img)


                    #region ----------------------- two methods to get the max_score class------------------------
                    if self.use_milvus:
                        ...
                    else:
                        cls_list  = []            # a face might be simliar as several faces in the dict
                        max_cls_scores = []       # for those faces, we have a score to indicate similarity

                        for cls in self.feat_dict:
                            cls_simi_scores = []  # the current box's similarity scores with the cls-th person's feature vec.
                            cls_list.append(cls)

                            for feat in self.feat_dict[cls]: # one person has several feature vec
                                '''
                                   refer to face_and_feature.jpg
                                   feat_dict has n persons
                                   each person has several feature vec
                                   So, we need two for loop. Outter for loop --> n 
                            
                                '''
                                simi_score = compare(torch.tensor(feature_test), torch.tensor(feat))
                                cls_simi_scores.append(simi_score)

                            max_cls_scores.append(max(cls_simi_scores)) # simply speaking, each candidate(cls) in the dict shows their best to compare against
                    #endregion ----------------------- two methods to get the max_score class ------------------------


                    #region --------------------------------Thresholding and Visulizing ---------------------------------------------
                    if max_cls_scores > 0.4:
                        print(f"similarity:", max(max_cls_scores))
                        index = np.argmax(max_cls_scores)
                        pred_cls = cls_list[index]

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, pred_cls, (x1, y1 - 2), 0, 3, [255, 255, 255], thickness = 3, lineType = cv2.LINE_AA)
                    else:
                        cv2.imshow("image", image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    #endregion --------------------------------Thresholding and Visulizing ---------------------------------------------

        def VideoRecognize(self, source):
            # recognize face in the video or videostream
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) # short for directshow via video input

            while True: # keep reading the video stream
                ret, frame = cap.read()

                if not ret:
                    continue

                t0 = time.time()
                bboxes = self.detector(frame)
                t1 = time.time()
                print(f"detection: ----> {t1 - t0}s")

                if bboxes:
                    for box in bboxes:
                        x1, y1, x2, y2 = box[0], box[1],box[2], box[3]
                        M = self.get_alignment_matrix(box)
                        aligned_croped_img = self.align(M, frame.copy())
                        aligned_croped_img = Image.fromarray(aligned_croped_img)
                        feature_test = self.extractor(aligned_croped_img)

                        # region ----------------------- two methods to get the max_score class------------------------
                        if self.use_milvus:
                            ...
                        else:
                            cls_list = []
                            max_cls_scores = []

                            for cls in self.feat_dict:
                                cls_simi_scores = []
                                cls_list.append(cls)

                                for feat in self.feat_dict[cls]:
                                    simi_score = compare(torch.tensor(feature_test), torch.tensor(feat))
                                    cls_simi_scores.append(simi_score)

                                max_cls_scores.append(max(cls_simi_scores))
                        # endregion ----------------------- two methods to get the max_score class ------------------------

                        # region --------------------------------Thresholding and Visulizing ---------------------------------------------
                        if max_cls_scores > 0.4:
                            print(f"similarity:", max(max_cls_scores))
                            index = np.argmax(max_cls_scores)
                            pred_cls = cls_list[index]

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, pred_cls, (x1, y1 - 2), 0, 3, [255, 255, 255], thickness=3,
                                        lineType=cv2.LINE_AA)



                        # endregion --------------------------------Thresholding and Visulizing ---------------------------------------------

                    cv2.imshow("1", frame)
                    t_end = time.time()
                    print(f"one frame ---> {t_end - t0}s \n")

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        # ------------------------------------------- 3 Major Functionalities ------------------------------------------
        #endregion

        #region ------------------------------------------- Face alignment -------------------------------------------
        def align(self, M, image): # todo    details about face alignment will be introduced in the next few days
            aligned_image = cv2.warpAffine(image, M, dsize=(128,128)) # 128 * 128 are the input size for feature extractor
            return aligned_image

        def get_alignment_matrix(self, box): #todo
            # find the mapping from landmarks in box to std_landmarks

            std_landmarks = np.array([ # std landmarks coordinates gain by arcface.
                 [40.979567061066064, 41.516426774567584],
                 [88.8882824975239, 40.93974246129967],
                 [66.18679549114333, 67.77924169228517],
                 [44.374469330991076, 89.78187673839847],
                 [86.25706338749815, 90.07465963987703]
            ], dtype = np.float32) #todo

            landmarks = np.array([
                 [box[4], box[5]],
                 [box[6], box[7]],
                 [box[8], box[9]],
                 [box[10],box[11]],
                 [box[12],box[13]],
            ], dtype = np.float32) #todo

            Matrix = self.__find_mapping_to_std_landmarks__(landmarks, std_landmarks)

            return Matrix

        def __find_mapping_to_std_landmarks__(self, landmarks, std_landmarks): #todo
            # construct Q matrix, 10 * 4
            Q = np.zeros((10, 4))

            # construct S matrix 10 * 1
            S = std_landmarks.reshape(-1, 1)

            for i in range(5): # 5 landmarks # Put the 5 points into the Q matirx
                x, y = landmarks[i]
                Q[i * 2 + 0] = x, y,  1, 0
                Q[i * 2 + 1] = y, -x, 0, 1

            M = (np.linalg.inv(Q.T @ Q) @ Q.T @ S).squeeze()

            matrix = np.array([
                [M[0],  M[1], M[2]],
                [-M[1], M[0], M[3]],
            ])

            return matrix
        #endregion ---------------------------------------- Face alignment ------------------------------------------




if __name__ == "main":
    # ------------------------------------------- arguments ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(r"--face_det_weight", type = str, default =
                        r"yolov5_face_master/best.pt")
    parser.add_argument(r"--feature_dict", type = str, default = r"arcface_pytorch\id_name.json"
                        , help= r"save several feature for each person")
    parser.add_argument(r"--save_crop_dir", type = str, default = r"arcface_pytorch\face_data"
                        , help= r"save some pictures for a given person")
    parser.add_argument("--source", type = str, defulat = 0, help = r"source of the input, video, image, rtsp, webcam etc") # if it is 0, 0 should be converted into int.
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

