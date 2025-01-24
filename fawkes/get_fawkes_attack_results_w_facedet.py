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


0.05 -> 0.5 ASR, 68, samples
"""

face_detector = YoloV5FaceDetector()
im_scorer = ImageScorer()
attack_directory = "final_w_facedet"
attack_folders = glob.glob(attack_directory + "/*")
attack_folders.sort()
prefix = "TEST"
necessary_files = ["cloaked_source_0.005.jpg", "cloaked_source_0.01.jpg", "cloaked_source_0.02.jpg",
                     "cloaked_source_0.03.jpg", "cloaked_source_0.04.jpg", 
                   "cloaked_source_0.05.jpg", "cloaked_source_0.06.jpg", "cloaked_source_0.07.jpg", 
                   "cloaked_source_0.08.jpg", "cloaked_source_0.09.jpg", "cloaked_source_0.1.jpg"]
                #    "cloaked_source_0.1.jpg", 
                #    "cloaked_source_0.2.jpg","cloaked_source_0.3.jpg",
                #     "cloaked_source_0.4.jpg", "cloaked_source_0.5.jpg"]
threshold = .88 # verilight decision threshold
rhos = [0.005, 0.01, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

def get_stats(rho):
    success = 0
    tot_samples = 0
    whole_imgs_dissims = []
    cropped_imgs_dissims = []
    for attack_folder in attack_folders:
        if ".csv" in attack_folder:
            continue
    
        source_path = attack_folder + "/source.jpg"
        cloak_path = attack_folder + f"/{prefix}_cloaked_source_{rho}.jpg"
        target_path = attack_folder + "/target.jpg"

        next_folder = False
        for n in necessary_files:
            if not os.path.exists(f"{attack_folder}/{prefix}_{n}"):
                # print("no ",f"{attack_folder}/{prefix}_{n}")
                next_folder = True
                break
        if next_folder:
            continue

        if not os.path.exists(source_path) or not os.path.exists(cloak_path) or not os.path.exists(target_path):
            continue

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
        if source_target_theta < threshold:
            continue
        source_cloak_theta = np.arccos(np.dot(source_emb, cloak_emb))
        cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
        if cloak_target_theta <  threshold:
            success += 1
        tot_samples += 1
        # print("Source-Target angle: ", source_target_theta)
        # print("Cloak-Target angle: ", cloak_target_theta)
        # print("Source-Cloak angle: ", source_cloak_theta)

    
        cloak_img = tf.image.decode_image(tf.io.read_file(cloak_path))
        source_img = tf.image.decode_image(tf.io.read_file(source_path))
        source_cloak_ssim =  tf.image.ssim(cloak_img, source_img, max_val=255.0)
        source_cloak_dissim = (1 - source_cloak_ssim)/2

        source_bb ,_, _, _ = face_detector.detect_in_image(source_path)
        cloak_bb, _, _, _ = face_detector.detect_in_image(cloak_path)
        source_bb = source_bb.astype(int)[0]
        cloak_bb = cloak_bb.astype(int)[0]
        max_bb = np.maximum(source_bb, cloak_bb)
        source_crop = source_img[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
        cloak_crop = cloak_img[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
        cropped_source_cloak_ssim =  tf.image.ssim(cloak_crop, source_crop, max_val=255.0)
        cropped_source_cloak_dissim = (1 - cropped_source_cloak_ssim)/2
        whole_imgs_dissims.append(source_cloak_dissim)
        cropped_imgs_dissims.append(cropped_source_cloak_dissim)
        # print(f"Full image dissimilarity: {source_cloak_dissim}. Cropped image dissimilarity: {cropped_source_cloak_dissim}")

        if tot_samples > 49:
            break
        
    print(f"Success rate: {success/tot_samples}. Whole img dissim: {np.mean(whole_imgs_dissims)}. Cropped img dissim: {np.mean(cropped_imgs_dissims)}. {tot_samples} samples considered.")


def aggregated_stats():
    res_dict = {}
    res_dict['0.0-0.01'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.01-0.02'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.02-0.03'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.03-0.04'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.04-0.05'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.05-0.06'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.06-0.07'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.07-0.08'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.08-0.09'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['0.09-0.1'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    res_dict['greater'] = {'successes': 0, 'total': 0, 'whole_imgs_dissims': [], 'cropped_imgs_dissims': []}
    for rho in rhos:
        for attack_folder in attack_folders:
            if ".csv" in attack_folder:
                continue
        
            source_path = attack_folder + "/source.jpg"
            cloak_path = attack_folder + f"/{prefix}_cloaked_source_{rho}.jpg"
            target_path = attack_folder + "/target.jpg"

            next_folder = False
            for n in necessary_files:
                if not os.path.exists(f"{attack_folder}/{prefix}_{n}"):
                    # print("no ",f"{attack_folder}/{prefix}_{n}")
                    next_folder = True
                    break
            if next_folder:
                continue

            if not os.path.exists(source_path) or not os.path.exists(cloak_path) or not os.path.exists(target_path):
                continue

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
            if source_target_theta < threshold:
                continue
            source_cloak_theta = np.arccos(np.dot(source_emb, cloak_emb))
            cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
         
        
            cloak_img = tf.image.decode_image(tf.io.read_file(cloak_path))
            source_img = tf.image.decode_image(tf.io.read_file(source_path))
            source_cloak_ssim =  tf.image.ssim(cloak_img, source_img, max_val=255.0)
            source_cloak_dissim = (1 - source_cloak_ssim)/2

            source_bb ,_, _, _ = face_detector.detect_in_image(source_path)
            cloak_bb, _, _, _ = face_detector.detect_in_image(cloak_path)
            source_bb = source_bb.astype(int)[0]
            cloak_bb = cloak_bb.astype(int)[0]
            max_bb = np.maximum(source_bb, cloak_bb)
            source_crop = source_img[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
            cloak_crop = cloak_img[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
            cropped_source_cloak_ssim =  tf.image.ssim(cloak_crop, source_crop, max_val=255.0)
            cropped_source_cloak_dissim = (1 - cropped_source_cloak_ssim)/2
            
            dissim_bin = None
            if cropped_source_cloak_dissim >= 0 and cropped_source_cloak_dissim <= 0.01:
                dissim_bin = '0.0-0.01'
            elif cropped_source_cloak_dissim > 0.01 and cropped_source_cloak_dissim <= 0.02:
                dissim_bin = '0.01-0.02'
            elif cropped_source_cloak_dissim > 0.02 and cropped_source_cloak_dissim <= 0.03:
                dissim_bin = '0.02-0.03'
            elif cropped_source_cloak_dissim > 0.03 and cropped_source_cloak_dissim <= 0.04:
                dissim_bin = '0.03-0.04'
            elif cropped_source_cloak_dissim > 0.04 and cropped_source_cloak_dissim <= 0.05:
                dissim_bin = '0.04-0.05'
            elif cropped_source_cloak_dissim > 0.05 and cropped_source_cloak_dissim <= 0.06:
                dissim_bin = '0.05-0.06'
            elif cropped_source_cloak_dissim > 0.06 and cropped_source_cloak_dissim <= 0.07:
                dissim_bin = '0.06-0.07'
            elif cropped_source_cloak_dissim > 0.07 and cropped_source_cloak_dissim <= 0.08:
                dissim_bin = '0.07-0.08'
            elif cropped_source_cloak_dissim > 0.08 and cropped_source_cloak_dissim <= 0.09:
                dissim_bin = '0.08-0.09'
            elif cropped_source_cloak_dissim > 0.09 and cropped_source_cloak_dissim <= 0.1:
                dissim_bin = '0.09-0.1'
            else:
                dissim_bin = 'greater'

            print(f"Bin: {dissim_bin}. Theta: {cloak_target_theta}")
            if cloak_target_theta <  threshold:
                res_dict[dissim_bin]['successes'] += 1

            res_dict[dissim_bin]['total'] += 1
            res_dict[dissim_bin]['whole_imgs_dissims'].append(source_cloak_dissim)
            res_dict[dissim_bin]['cropped_imgs_dissims'].append(cropped_source_cloak_dissim)
        
        with open("aggregated_stats_temp.csv", "w") as f:
            f.write('bin,succes_rate,successes,total,whole_imgs_dissim,cropped_imgs_dissim\n')
            for k, v in res_dict.items():
                if v['total'] == 0:
                    success_rate = 0
                else:
                    success_rate = v['successes']/v['total']
                f.write(f"{k},{success_rate},{v['successes']},{v['total']},{np.mean(v['whole_imgs_dissims'])},{np.mean(v['cropped_imgs_dissims'])}\n")


    with open("aggregated_stats.csv", "w") as f:
        f.write('bin,succes_rate,successes,total,whole_imgs_dissim,cropped_imgs_dissim\n')
        for k, v in res_dict.items():
            if v['total'] == 0:
                success_rate = 0
            else:
                success_rate = v['successes']/v['total']
            f.write(f"{k},{success_rate},{v['successes']},{v['total']},{np.mean(v['whole_imgs_dissims'])},{np.mean(v['cropped_imgs_dissims'])}\n")

# for rho in rhos:
#     print(f"Rho: {rho}")
#     get_stats(rho)
#     print("\n")

aggregated_stats()