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
from differentiation_w_facedet import FawkesMaskGeneration
from utils import init_gpu, dump_image, reverse_process_cloaked, \
    Faces, load_image, load_extractor

from fawkes.align_face import aligner


# from differntiator.py
def resize_tensor(input_tensor, model_input_shape):
    if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
        return input_tensor
    resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
    return resized_tensor

def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X


IMG_SIZE = 300
PREPROCESS = 'raw'
INPUT_PREPROCESS = 'resnet_arcface'

    
class DetRecModel(object):
    """
    Only works with one image
    """
    def __init__(self, det_model, rec_model):
        self.det_model = det_model
        self.rec_model = rec_model

    def predict(self, imgs):
        imgs = imgs[0, :, :, :]
        bbs, pps, ccs = self.det_model(imgs, numpy=False) # should be RGB
        bbs = bbs[0]
        x1, y1, x2, y2 = bbs
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped = imgs[y1:y2, x1:x2]
        cropped = (cropped - 127.5) * 0.0078125
        cropped = resize_tensor(cropped, (112, 112, 3))
        cropped = tf.expand_dims(cropped, axis=0) # add batch dimension
        embeds = self.rec_model(cropped)
        return embeds

    def __call__(self, x):
        return self.predict(x)


class Fawkes(object):
    def __init__(self, feature_extractor, gpu, batch_size, mode="custom", th = None, max_step = None, lr = None):

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


        self.protector = None
        self.protector_param = None
        self.feature_extractors_ls = []
        det_model = YoloV5FaceDetector()
        rec_model = tf.keras.models.load_model("r50_magface_MS1MV2.h5", compile=False)
        det_rec_model = DetRecModel(det_model, rec_model)
        self.feature_extractors_ls.append(det_rec_model)
          
    def run_protection(self, img_path, target_img_path, th=0.04, sd=1e7, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False, exp="", maximize=False,
                       save_last_on_failed=True):

        current_param = "-".join([str(x) for x in [self.th, sd, self.lr, self.max_step, batch_size, format,
                                                   separate_target, debug]])
        
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.imread(target_img_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
        target_img = cv2.resize(target_img, (IMG_SIZE, IMG_SIZE))

        original_images = np.array([original_img])
        target_images = np.array([target_img])

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

        return protected_images
        

def gen_identity_pairs(num_identities, results_directory):
 
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

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

    for identity_pair in identity_pairs:
        source_img_paths = glob.glob(identity_pair[0] + "/*")
        target_img_paths = glob.glob(identity_pair[1] + "/*")
        source_name = os.path.basename(identity_pair[0])
        target_name = os.path.basename(identity_pair[1])
        # print("Processing ", identity_pair)

        # randomly choose one of the images
        source_img_path = np.random.choice(source_img_paths, 1, replace=False)[0]
        target_img_path = np.random.choice(target_img_paths, 1, replace=False)[0]
        
        try:
            source_img = cv2.imread(source_img_path)
            target_img = cv2.imread(target_img_path)
        except:
            continue

        # make directory
        os.makedirs(f"{results_directory}/{source_name}2{target_name}", exist_ok=True)
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/source.jpg", source_img)
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/target.jpg", target_img)

def run_test(perturbation_budget, results_directory):

    feature_extractors = ["resnet_arcface"]
    gpu = '0'
    th = perturbation_budget 
    max_step = 150
    sd = 1e6
    lr = 10
    batch_size = 1 
    format = "jpeg"
    separate_target = True
    debug = True
    no_align = False

    victim_directories = glob.glob(results_directory + "/*")
    f = open(f"{results_directory}/cloaking_log.csv", "w")
    f.write("source,target\n")
    for dir in victim_directories:
        source_img_path = dir + "/source.jpg"
        target_img_path = dir + "/target.jpg"
        source_name = os.path.basename(dir).split("2")[0]
        target_name = os.path.basename(dir).split("2")[1]
        source_img = cv2.imread(source_img_path)

        if os.path.exists(f"{results_directory}/{source_name}2{target_name}/TEST_cloaked_source_{perturbation_budget}.jpg"):
            continue
        
        # create a new protector specific to this identity/experiment
        protector = Fawkes(feature_extractors, gpu, batch_size, mode="custom", th=th, max_step=max_step, lr=lr) # custom allows us to specify our own DSSIM threshold 

        print(Fore.MAGENTA + f"Cloaking {source_name} to {target_name} with rho {perturbation_budget}" + Style.RESET_ALL)
        res = protector.run_protection(source_img_path, target_img_path, th=th, sd=sd, lr=lr,
                            max_step=max_step,
                            batch_size=batch_size, format=format,
                            separate_target=separate_target, debug=debug, no_align=no_align)
        if type(res) == int:
            print("No face or more than one face detected. Skipping")
            continue
        protected_img = res
        protected_img = protected_img[0]
        protected_img = protected_img.astype(np.uint8)
        protected_img = cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR)
        original_source_size = source_img.shape[:2]
        protected_img = cv2.resize(protected_img, (original_source_size[1], original_source_size[0]))
        
        cv2.imwrite(f"{results_directory}/{source_name}2{target_name}/TEST_cloaked_source_{perturbation_budget}.jpg", protected_img)
        f.write(f"{source_name},{target_name}\n")
        f.flush()

    f.close()

    # protected_img = protector.run_protection(img_path, th=th, sd=sd, lr=lr,
    #                          max_step=max_step,
    #                          batch_size=batch_size, format=format,
    #                          separate_target=separate_target, debug=debug, no_align=no_align)
    # cv2.imshow("Protected Image", protected_img)
    # cv2.waitKey(0)

def perturb_single_img(img_path, target_img_path, perturbation_budget, results_directory):
    feature_extractors = ["resnet_arcface"]
    gpu = '0'
    th = perturbation_budget 
    max_step = 100
    sd = 1e6
    lr = 10
    batch_size = 1 
    format = "jpeg"
    separate_target = True
    debug = True
    no_align = False
    # create a new protector specific to this identity/experiment
    protector = Fawkes(feature_extractors, gpu, batch_size, mode="custom", th=th, max_step=max_step, lr=lr) # custom allows us to specify our own DSSIM threshold 

    og_img_size = cv2.imread(img_path).shape[:2]
    print(Fore.MAGENTA + f"Cloaking {img_path} to {target_img_path} with rho {perturbation_budget}" + Style.RESET_ALL)
    res = protector.run_protection(img_path, target_img_path, th=th, sd=sd, lr=lr,
                            max_step=max_step,
                            batch_size=batch_size, format=format,
                            separate_target=separate_target, debug=debug, no_align=no_align)
    if res is None:
        print("ERROR")

    protected_img = res[0]
    protected_img = protected_img.astype(np.uint8)
    protected_img = cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR)
    protected_img = cv2.resize(protected_img, (og_img_size[1], og_img_size[0]))
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    cv2.imwrite(f"{results_directory}/cloaked_source_{perturbation_budget}.jpg", protected_img)


class ImageScorer():
    def __init__(self):
        # self.extractor = load_extractor(extractor_name)
        # self.extractor = buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
        self.extractor = tf.keras.models.load_model("r50_magface_MS1MV2.h5", compile=False)
        self.aligner = aligner()
        yolo_detector = YoloV5FaceDetector()
        rec_model = tf.keras.models.load_model("r50_magface_MS1MV2.h5", compile=False)
        self.det_rec_model = DetRecModel(yolo_detector, rec_model)

    def get_embedding(self, img_path):
        # do exact same processing as done on images provided to Fawkes
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.array([img])
        emb = self.det_rec_model.predict(img)
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


