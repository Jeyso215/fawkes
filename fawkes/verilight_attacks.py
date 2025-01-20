#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import argparse
import glob
import logging
import os
import sys
import cv2
from keras.preprocessing import image
from tf_insightface import buildin_models
from tf_yolo import YoloV5FaceDetector
from colorama import Style, Fore
import os, contextlib

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import numpy as np
from differentiator import FawkesMaskGeneration
from utils import init_gpu, dump_image, reverse_process_cloaked, \
    Faces, load_image, load_extractor

from fawkes.align_face import aligner


def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X


IMG_SIZE = 112
PREPROCESS = 'raw'
INPUT_PREPROCESS = 'resnet_arcface'

    
class YoloFaces(object):
    def __init__(self):
        self.original_images = []
        self.cropped_faces = []
        self.bbs = []
        self.cropped_cloaks = []
    def merge_faces(self):
        cloaked_images = []
        for i in range(len(self.cropped_cloaks)):
            bbox = self.bbs[i]
            x1, y1, x2, y2 = bbox.astype(int)
            width = x2 - x1
            height = y2 - y1
            og_img = self.original_images[i]    
            cloaked_img = self.cropped_cloaks[i]
            # cloaked_img = cloaked_img.astype(np.uint8)
            # cloaked_img = cv2.cvtColor(cloaked_img, cv2.COLOR_BGR2RGB)
            # print("cloaked shape: ", cloaked_img.shape)
            # cv2.imshow("cloaked", cloaked_img)
            # cv2.waitKey(0)
            cloaked_img = cv2.resize(cloaked_img, (width, height))
            # cv2.imshow("Face", og_img)
            # cv2.waitKey(0)
            # cv2.imshow("Cloaked Face", cloaked_faces[i])
            # cv2.waitKey(0)
            og_img[y1:y2, x1:x2] = cloaked_img
            print(np.max(og_img), np.min(og_img), np.max(cloaked_img), np.min(cloaked_img))
            # og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
            cloaked_images.append(og_img)

        return cloaked_images

    

class Fawkes(object):
    def __init__(self, feature_extractor, gpu, batch_size, mode="custom", th = None, max_step = None, lr = None,
                 aligner_type = "Yolo"):

        self.feature_extractor = feature_extractor
        self.gpu = gpu
        self.batch_size = batch_size
        self.mode = mode
        if mode == "custom":
            if th is None or max_step is None or lr is None:
                raise Exception("th, max_step, lr must be specified in custom mode")
            extractors = feature_extractor
        else:
            th, max_step, lr, extractors = self.mode2param(self.mode)
        self.th = th
        self.lr = lr
        self.max_step = max_step
        if gpu is not None:
            init_gpu(gpu)

        self.aligner_type = aligner_type
        if aligner_type == "Yolo":
            self.yolo_detector = YoloV5FaceDetector()
        self.aligner = aligner()

        self.protector = None
        self.protector_param = None
        self.feature_extractors_ls = []
        for name in extractors:
            if name == "resnet_arcface":
                # model = buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
                model = tf.keras.models.load_model("r50_magface_MS1MV2.h5", compile=False)
                self.feature_extractors_ls.append(model)
            else:
                self.feature_extractors_ls.append(load_extractor(name))

    def mode2param(self, mode):
        if mode == 'low':
            th = 0.004
            max_step = 40
            lr = 25
            extractors = ["extractor_2"]

        elif mode == 'mid':
            th = 0.012
            max_step = 75
            lr = 20
            extractors = ["extractor_0", "extractor_2"]

        elif mode == 'high':
            th = 0.017
            max_step = 150
            lr = 15
            extractors = ["extractor_0", "extractor_2"]

        else:
            raise Exception("mode must be one of 'min', 'low', 'mid', 'high'")
        return th, max_step, lr, extractors
    
    def get_crop_align_face(self, img_path, no_align=False):
        if self.aligner_type == "Yolo":
            face_obj = YoloFaces()
            img = cv2.imread(img_path)
            bbs, _, ccs, nimgs = self.yolo_detector.detect_in_image(img, image_format="BGR")
            x1, y1, x2, y2 = bbs[0].astype(int)
            # print(x1-x2, y1-y2, nimgs[0].shape)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow("Face", img)
            # cv2.waitKey(0)
            my_cropped = img[y1:y2, x1:x2]
            my_cropped = cv2.resize(my_cropped, (112, 112))
            my_cropped = cv2.cvtColor(my_cropped, cv2.COLOR_BGR2RGB) # model expects RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Face", my_cropped)
            # cv2.waitKey(0)
            # print(my_cropped.shape)
            # resize
            face_obj.cropped_faces = [my_cropped]
            face_obj.bbs = bbs
            face_obj.original_images = [img] 
            return face_obj
        if self.aligner_type == "no_crop":
            img = cv2.imread(img_path)
            bbox = [0, 0, img.shape[1], img.shape[0]]
            bbox = np.array(bbox)
            my_cropped = cv2.resize(img, (112, 112))
            my_cropped = cv2.cvtColor(my_cropped, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_obj = YoloFaces()
            face_obj.cropped_faces = [my_cropped]
            face_obj.bbs = [bbox]
            face_obj.original_images = [img]
            return face_obj
        else:
            img = load_image(img_path)
            image_paths = [img_path]
            loaded_images = [img]
            faces = Faces(image_paths, loaded_images, self.aligner, verbose=1, no_align=no_align)
            return faces
    


    def run_protection(self, img_path, target_img_path, th=0.04, sd=1e7, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False, exp="", maximize=False,
                       save_last_on_failed=True):

        current_param = "-".join([str(x) for x in [self.th, sd, self.lr, self.max_step, batch_size, format,
                                                   separate_target, debug]])
        
        original_faces = self.get_crop_align_face(img_path, no_align=no_align)
        original_images = original_faces.cropped_faces
        if len(original_images) == 0:
            print("No face detected. ")
            return 2
        if len(original_images) > 1:
            print("More than one face detected. ")
            return 2
        # cv2.imshow("Face", original_images[0][:, :, ::-1])
        # cv2.waitKey(0)
        original_images = np.array(original_images)
     

        target_faces = self.get_crop_align_face(target_img_path, no_align=no_align)
        target_images = target_faces.cropped_faces
        if len(target_images) == 0:
            print("No target face detected. ")
            return 2
        if len(target_images) > 1:
            print("More than one target face detected. ")
            return 2
        # cv2.imshow("Face", target_images[0][:, :, ::-1])
        # cv2.waitKey(0)
        target_images = np.array(target_images)

        if current_param != self.protector_param:
            self.protector_param = current_param
            if self.protector is not None:
                del self.protector
            if batch_size == -1:
                batch_size = len(original_images)
            self.protector = FawkesMaskGeneration(self.feature_extractors_ls,
                                                  batch_size=batch_size,
                                                  mimic_img=True,
                                                  intensity_range=PREPROCESS,
                                                  input_preprocesing = INPUT_PREPROCESS,
                                                  initial_const=sd,
                                                  learning_rate=self.lr,
                                                  max_iterations=self.max_step,
                                                  l_threshold=self.th,
                                                  verbose=debug,
                                                  maximize=maximize,
                                                  keep_final=False,
                                                  image_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  loss_method='features',
                                                  tanh_process=True,
                                                  save_last_on_failed=save_last_on_failed,
                                                  )
        protected_images = generate_cloak_images(self.protector, original_images, target_emb = target_images)


        
        if self.aligner_type == "Yolo" or self.aligner_type == "no_crop":
            original_faces.cropped_cloaks = protected_images.astype(np.uint8)
            final_images = original_faces.merge_faces()
            p_img = final_images[0]
        else:
            original_faces.cloaked_cropped_faces = protected_images
            final_images, images_without_face = original_faces.merge_faces(
                reverse_process_cloaked(protected_images, preprocess=PREPROCESS),
                reverse_process_cloaked(original_images, preprocess=PREPROCESS))
            p_img = final_images[0]
        return p_img, original_faces, target_faces


def run_test(num_identities, perturbation_budget, results_directory, val = False):
    if val:
        img_scorer = ImageScorer()

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

    feature_extractors = ["resnet_arcface"]
    gpu = '0'
    th = perturbation_budget 
    max_step = 100
    sd = 1e6
    lr = 20
    batch_size = 1 
    format = "jpeg"
    separate_target = True
    debug = False
    no_align = False

    lfw_root = "lfw"
    lfw_identities = glob.glob(lfw_root + "/*")
    candidate_identities = []
    for identity in lfw_identities:
        identity_img_paths = glob.glob(identity + "/*")
        if len(identity_img_paths) >= 3:
            candidate_identities.append(identity)
    identity_pairs = []
    for i in range(num_identities):
        id1 = np.random.randint(0, len(candidate_identities))
        id2 = np.random.randint(0, len(candidate_identities))
        while id1 == id2:
            id2 = np.random.randint(0, len(candidate_identities))
        identity_pairs.append((candidate_identities[id1], candidate_identities[id2]))

    os.makedirs(results_directory, exist_ok=True)
    f = open(f"{results_directory}/cloaking_log.csv", "w")
    f.write("source,target\n")
    for identity_pair in identity_pairs:
        source_img_paths = glob.glob(identity_pair[0] + "/*")
        target_img_paths = glob.glob(identity_pair[1] + "/*")
        source_name = os.path.basename(identity_pair[0])
        target_name = os.path.basename(identity_pair[1])
        # print("Processing ", identity_pair)

        # randomly choose one of the images
        source_img_path = np.random.choice(source_img_paths, 1, replace=False)[0]
        target_img_path = np.random.choice(target_img_paths, 1, replace=False)[0]
        if val:
            # get the other source images 
            other_source_imgs = []
            for img_path in source_img_paths:
                if img_path != source_img_path:
                    try:
                        source_img = cv2.imread(img_path)
                        if source_img is None:
                            continue
                    except:
                        continue
                    other_source_imgs.append(img_path)
        try:
            source_img = cv2.imread(source_img_path)
            target_img = cv2.imread(target_img_path)
        except:
            continue

        # create a new protector specific to this identity/experiment
        protector = Fawkes(feature_extractors, gpu, batch_size, mode="custom", th=th, max_step=max_step, lr=lr, 
                           aligner_type="Yolo") # custom allows us to specify our own DSSIM threshold 


        print(Fore.MAGENTA + f"Cloaking {source_name} to {target_name}" + Style.RESET_ALL)
        res = protector.run_protection(source_img_path, target_img_path, th=th, sd=sd, lr=lr,
                            max_step=max_step,
                            batch_size=batch_size, format=format,
                            separate_target=separate_target, debug=debug, no_align=no_align)
        if type(res) == int:
            print("No face or more than one face detected. Skipping")
            continue
        protected_img, original_faces, target_faces = res
        protected_img = cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR)
        os.makedirs(f"{results_directory}/{source_name}2{target_name}", exist_ok=True)
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/source.jpg", source_img)
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/target.jpg", target_img)
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/cloaked_source.jpg", protected_img)

        # save crops for computign results of attack later
        for i in range(len(original_faces.cropped_faces)):
            og_face = cv2.cvtColor(original_faces.cropped_faces[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/source_crop.jpg", og_face)
            cropped_cloak = cv2.cvtColor(original_faces.cropped_cloaks[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/cloak_crop.jpg", cropped_cloak)
        for i, face in enumerate(target_faces.cropped_faces):
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/target_crop.jpg", face)

        f.write(f"{source_name},{target_name}\n")
        f.flush()

        if val:
            print("Scoring images")
            img_scorer.score_images(source_img_path, f"{results_directory}/{source_name}2{target_name}/cloaked_source.jpg",
                                     target_img_path,
                                    og_references=other_source_imgs)

    f.close()

    # protected_img = protector.run_protection(img_path, th=th, sd=sd, lr=lr,
    #                          max_step=max_step,
    #                          batch_size=batch_size, format=format,
    #                          separate_target=separate_target, debug=debug, no_align=no_align)
    # cv2.imshow("Protected Image", protected_img)
    # cv2.waitKey(0)

# from differntiator.py
def resize_tensor(input_tensor, model_input_shape):
    if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
        return input_tensor
    resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
    return resized_tensor

class ImageScorer():
    def __init__(self):
        # self.extractor = load_extractor(extractor_name)
        # self.extractor = buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
        self.extractor = tf.keras.models.load_model("r50_magface_MS1MV2.h5", compile=False)
        self.aligner = aligner()
        self.yolo_detector = YoloV5FaceDetector()

    def get_embedding(self, img_path, detect_face = True):
        # img = load_image(img_path)
        # with open(os.devnull, 'w') as devnull:
        #     with contextlib.redirect_stdout(devnull):
        #         faces = Faces([img_path], [img], self.aligner, verbose=0)
        # img = faces.cropped_faces
        img = cv2.imread(img_path)
        if detect_face:
            bbs, _, ccs, nimgs = self.yolo_detector.detect_in_image(img, image_format="BGR")
        else:
            nimgs = np.array([img[:, :, ::-1]])
        if len(nimgs) == 0:
            print("No face detected. ")
            return None
        if len(nimgs) > 1:
            print("More than one face detected. ")
            return None
        img = nimgs
        img = tf.Variable(img, dtype=np.float32)
        img = resize_tensor(img, (112, 112, 3))
        img = (img- 127.5) * 0.0078125 # if using resnet cosfcae
        emb = self.extractor(img)
        emb = emb.numpy()
        emb = emb[0]
        emb = emb / np.linalg.norm(emb) 
        return emb

    def score_images(self, og_img_path, protected_img_path, target_img_path, og_references = []):
       
        og_emb = self.get_embedding(og_img_path)
        protected_emb = self.get_embedding(protected_img_path) # already saved the detected version
        target_emb = self.get_embedding(target_img_path)
        if og_emb is None or protected_emb is None or target_emb is None:
            return
        og_prot_cos_angle = np.arccos(np.dot(og_emb, protected_emb))
        prot_target_angle = np.arccos(np.dot(protected_emb, target_emb))
        og_target_angle = np.arccos(np.dot(og_emb, target_emb))
        print(f"OG-Target Angle (rad): {og_target_angle}")
        print(f"Cloaked-Target Angle (rad): {prot_target_angle}")
        print(f"OG-Cloaked Angle (rad): {og_prot_cos_angle}")
        for i, ref in enumerate(og_references):
            ref_emb = self.get_embedding(ref)
            if ref_emb is None:
                continue
            ref_prot_angle = np.arccos(np.dot(ref_emb, og_emb))
            print(f"Reference-OG {i} Angle (rad): {ref_prot_angle}")

