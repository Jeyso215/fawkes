import glob
from verilight_attacks_w_facedet import ImageScorer
from colorama import Fore, Style
import numpy as np
import tensorflow as tf
import os 
rho = 0.001
im_scorer = ImageScorer()
attack_directory = "final_w_facedet"
attack_folders = glob.glob(attack_directory + "/*")
success = 0
for attack_folder in attack_folders:
    if ".csv" in attack_folder:
        continue
   
  
    source_path = attack_folder + "/source.jpg"
    cloak_path = attack_folder + f"/cloaked_source_{rho}.jpg"
    target_path = attack_folder + "/target.jpg"
    if not os.path.exists(source_path) or not os.path.exists(cloak_path) or not os.path.exists(target_path):
        continue

    print(Fore.MAGENTA + attack_folder + Style.RESET_ALL)
    source_emb = im_scorer.get_embedding(source_path)
    cloak_emb = im_scorer.get_embedding(cloak_path)
    target_emb = im_scorer.get_embedding(target_path)

    if source_emb is None or cloak_emb is None or target_emb is None:
        # print(Fore.RED + f"{source_emb}, {cloak_emb}, {target_emb}" + Style.RESET_ALL)
        nones  = []
        if source_emb is None:
            nones.append("source")
        if cloak_emb is None:
            nones.append("cloak")
        if target_emb is None:
            nones.append("target")
        print(Fore.RED + f"None embeddings for {nones}" + Style.RESET_ALL)
        continue
    source_target_theta = np.arccos(np.dot(source_emb, target_emb))
    if source_target_theta < 1.21:
        continue
    source_cloak_theta = np.arccos(np.dot(source_emb, cloak_emb))
    cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
    if cloak_target_theta <  1.21:
        success += 1
    print("Source-Target angle: ", source_target_theta)
    print("Cloak-Target angle: ", cloak_target_theta)
    print("Source-Cloak angle: ", source_cloak_theta)

 
    cloak_img = tf.image.decode_image(tf.io.read_file(cloak_path))
    source_img = tf.image.decode_image(tf.io.read_file(source_path))
    source_cloak_ssim =  tf.image.ssim(cloak_img, source_img, max_val=255.0)
    source_cloak_dissim = (1 - source_cloak_ssim)/2
    print("Facial region DISSIM: ", source_cloak_dissim)

print("Success rate: ", success/(len(attack_folders) - 1))


