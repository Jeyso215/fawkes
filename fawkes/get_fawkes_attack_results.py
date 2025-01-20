import glob
from verilight_experiments import ImageScorer
from colorama import Fore, Style
import numpy as np
import tensorflow as tf

on_crop = False

im_scorer = ImageScorer()
attack_directory = "verilight_tests"
attack_folders = glob.glob(attack_directory + "/*")
for attack_folder in attack_folders:
    print(Fore.MAGENTA + attack_folder + Style.RESET_ALL)
    if on_crop:
        source_path = attack_folder + "/source_crop.jpg"
        cloak_path = attack_folder + "/cloak_crop.jpg"
        target_path = attack_folder + "/target_crop.jpg"
        source_emb = im_scorer.get_embedding(source_path, detect_face=False)
        cloak_emb = im_scorer.get_embedding(cloak_path, detect_face=False)
        target_emb = im_scorer.get_embedding(target_path, detect_face=False)
    else:
        source_path = attack_folder + "/source.jpg"
        cloak_path = attack_folder + "/cloaked_source.jpg"
        target_path = attack_folder + "/target.jpg"
        source_emb = im_scorer.get_embedding(source_path, detect_face=True)
        cloak_emb = im_scorer.get_embedding(cloak_path, detect_face=True)
        target_emb = im_scorer.get_embedding(target_path, detect_face=True)


    source_target_theta = np.arccos(np.dot(source_emb, target_emb))
    source_cloak_theta = np.arccos(np.dot(source_emb, cloak_emb))
    cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
    print("Source-Target angle: ", source_target_theta)
    print("Cloak-Target angle: ", cloak_target_theta)
    print("Source-Cloak angle: ", source_cloak_theta)
    cloak_img = tf.image.decode_image(tf.io.read_file(cloak_path))
    source_img = tf.image.decode_image(tf.io.read_file(source_path))
    source_cloak_ssim =  tf.image.ssim(cloak_img, source_img, max_val=255.0)
    source_cloak_dissim = (1 - source_cloak_ssim)/2
    print("DISSIM: ", source_cloak_dissim)



