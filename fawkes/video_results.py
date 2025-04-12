import glob
from verilight_attacks_w_facedet import ImageScorer
from colorama import Fore, Style
import numpy as np
import tensorflow as tf
import os 
from tf_yolo import YoloV5FaceDetector
import cv2

"""
Results


"""

im_scorer = ImageScorer()
attack_directory = "vox/attack"
attack_folders = glob.glob(attack_directory + "/*")
attack_folders.sort()

threshold = 0.88

def get_stats(rho):
    """
    Get stats by budget and not actual resultant DSSIM
    """
    attack_successes = 0
    tot_samples = 0

    for attack_folder in attack_folders:
        source_frames_path = attack_folder + "/source_frames"
        cloaked_frames_path = attack_folder + f"/cloaked_frames_{rho}"
        if os.path.exists(cloaked_frames_path) == False:
            # print(f"{Fore.RED} Cloaked frames not found {Style.RESET_ALL}")
            continue
        else:
            print(Fore.BLUE + f"Checking {attack_folder}" + Style.RESET_ALL)
        target_path = attack_folder + "/target.jpg"
        target_emb = im_scorer.get_embedding(target_path)
        cloaked_frames = glob.glob(cloaked_frames_path + "/*")
        cloaked_frames.sort()
        frame_successes = [0 for _ in range(len(cloaked_frames))]
        for i, cloak_path in enumerate(cloaked_frames):

            cloak_emb = im_scorer.get_embedding(cloak_path)
    
            cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
            if cloak_target_theta <  threshold:
                frame_successes[i] = 1
            else:
                print("Frame attack failure!")

        tot_samples += 1
        if np.sum(frame_successes) == len(frame_successes):
            attack_successes += 1
        
    print(f"Success rate: {attack_successes/tot_samples}.")



get_stats(0.017)