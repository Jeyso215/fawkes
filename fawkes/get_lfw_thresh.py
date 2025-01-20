from tf_insightface import buildin_models
from verilight_attacks import ImageScorer
import glob
import numpy as np
import cv2
import itertools
import os
import pickle

def get_embeddings():
    im_scorer = ImageScorer()
    lfw_folders = glob.glob("lfw/*")
    os.makedirs("lfw_embs_yolo_take2", exist_ok=True)
    for folder_num, folder in enumerate(lfw_folders):
        print(f"Processing folder {folder_num}/{len(lfw_folders)}")
        person_name = folder.split("/")[-1]
        person_emb_folder = "lfw_embs_yolo_take2/" + person_name
        files = glob.glob(folder + "/*")
        if len(files) < 2:
            continue
        if os.path.exists(person_emb_folder):
            continue
        else:
            os.makedirs(person_emb_folder)
        id_embs = []
        for file in files:
            try:
                img = cv2.imread(file)
                if img is None:
                    continue
            except Exception as e:
                print(e)
                continue
            emb = im_scorer.get_embedding(file)
            if emb is None:
                continue
            id_embs.append(emb)
            with open(person_emb_folder + "/" + file.split("/")[-1] + ".pkl", "wb") as f:
                pickle.dump(emb, f)

def compute_same_id_scores():
    same_id_thetas = []
    lfw_folders = glob.glob("lfw_embs_yolo_take2/*")
    for folder_num, folder in enumerate(lfw_folders):
        print(f"Processing folder {folder_num}/{len(lfw_folders)}")
        files = glob.glob(folder + "/*")
        id_embs = []
        for file in files:
            with open(file, "rb") as f:
                emb = pickle.load(f)
            id_embs.append(emb)
        print("---------" + folder + "---------")
        emb_pairs = list(itertools.combinations([i for i in range(len(id_embs))], 2))
     
        for i, j in emb_pairs:
            theta = np.arccos(np.dot(id_embs[i], id_embs[j]))
            print(theta)
            same_id_thetas.append(theta)
        print("--------------------------------")
    print("average same theta: ", np.mean(same_id_thetas))
    with open("same_id_thetas_yolo_take2.pkl", "wb") as f:
        pickle.dump(same_id_thetas, f)
    
def compute_diff_id_scores():
    diff_id_thetas = []
    lfw_folders = glob.glob("lfw_embs_yolo_take2/*")
    id_combs = list(itertools.combinations([i for i in range(len(lfw_folders))], 2))
    for id_comb in id_combs:
        id1 = id_comb[0]
        id2 = id_comb[1]
        print(f"-------Processing {id1} and {id2} --------")
        files1 = glob.glob(lfw_folders[id1] + "/*")
        files2 = glob.glob(lfw_folders[id2] + "/*")
        for file1 in files1:
            with open(file1, "rb") as f:
                emb1 = pickle.load(f)
            for file2 in files2:
                with open(file2, "rb") as f:
                    emb2 = pickle.load(f)
                theta = np.arccos(np.dot(emb1, emb2))
                print(theta)
                diff_id_thetas.append(theta)
        print("--------------------------------")
    print("Average diff theta: ", np.mean(diff_id_thetas))
    with open("diff_id_thetas_yolo_take2.pkl", "wb") as f:
        pickle.dump(diff_id_thetas, f)

def get_get_bal_acc(same_X, diff_X, thresh):
    """
    Here 1 corresponds to "same person".
    False positive is therefore two embeddings of different people incorrectly verified as same
    """
  
    true_positives = 0 #diffs correctly classified as diffs
    false_positives = 0#sames classified as diff
    false_negatives = 0#diffs (deepfakes) falsely categorized as same
    true_negatives = 0 #sames correctly classified as sames

  
    for i in range(diff_X.shape[0]):
        if diff_X[i] > thresh:
            true_negatives += 1
        else:
            false_positives += 1
    for i in range(same_X.shape[0]):
        if same_X[i] > thresh:
            false_negatives += 1
        else:
            true_positives += 1
    # print(f"--------------- {thresh} --------------- ")
    # print(f"False positives: {false_positives}/{diff_X.shape[0]} diffs.")
    # print(f"True positives: {true_positives}/{same_X.shape[0]} sames.")
    # print(f"False negatives: {false_negatives}/{same_X.shape[0]} sames.")
    # print(f"True negatives: {true_negatives}/{diff_X.shape[0]} diffs.")
    # tpr = true_positives / (true_positives + false_negatives)
    # fpr = false_positives / (false_positives + true_negatives)
    # print(f"-------------------------------- ")
    bal_acc = 0.5 * ((true_positives / (true_positives + false_negatives)) + (true_negatives / (true_negatives + false_positives)))
    print(f"Thresh: {thresh}. Bal acc: {bal_acc}")
    return bal_acc



def get_best_threhsold():
    diff_dists_path = f"diff_id_thetas_yolo_take2.pkl" 
    same_dists_path = f"same_id_thetas_yolo_take2.pkl"

    with open(diff_dists_path, "rb") as pklfile:
        diff_dists = pickle.load(pklfile)
    with open(same_dists_path, "rb") as pklfile:
        same_dists = pickle.load(pklfile)

    diff_dists = np.array(diff_dists)
    same_dists = np.array(same_dists)

    best_bal_acc = float('-inf')
    best_bal_acc_t = None
    for t in range(100, int(max(np.max(same_dists), np.max(diff_dists))*100)):
        t /= 100
        bal_acc = get_get_bal_acc(same_dists, diff_dists, t)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_bal_acc_t = t
    print("Best bal acc: ", best_bal_acc)
    print("Best bal acc t: ", best_bal_acc_t)


# get_embeddings()
# compute_same_id_scores()
# compute_diff_id_scores()
get_best_threhsold()

