import tensorflow as tf
from verilight_attacks_w_facedet import ImageScorer
import numpy as np
from tf_yolo import YoloV5FaceDetector



im_scorer = ImageScorer()
face_detector = YoloV5FaceDetector()

test_case = "Joschka_Fischer2Valentino_Rossi"
src_path = f"faces_copy/{test_case}/source.jpg"
target_path = f"faces_copy/{test_case}/target.jpg"
cloaked_path = f"faces_copy/{test_case}/TEST_cloaked_source_0.1.jpg"


src_tensor = tf.image.decode_image(tf.io.read_file(src_path))
cloaked_tensor = tf.image.decode_image(tf.io.read_file(cloaked_path))
ssim = tf.image.ssim(src_tensor, cloaked_tensor, max_val=255)
dssim = (1 - ssim)/2
print(dssim)

source_emb = im_scorer.get_embedding(src_path)
cloak_emb = im_scorer.get_embedding(cloaked_path)
target_emb = im_scorer.get_embedding(target_path)
cloak_target_theta = np.arccos(np.dot(cloak_emb, target_emb))
print(cloak_target_theta)



source_bb ,_, _, _ = face_detector.detect_in_image(src_path)
cloak_bb, _, _, _ = face_detector.detect_in_image(cloaked_path)
source_bb = source_bb.astype(int)[0]
cloak_bb = cloak_bb.astype(int)[0]
max_bb = np.maximum(source_bb, cloak_bb)
source_crop = src_tensor[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
cloak_crop = cloaked_tensor[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
cropped_source_cloak_ssim =  tf.image.ssim(cloak_crop, source_crop, max_val=255.0)
cropped_source_cloak_dissim = (1 - cropped_source_cloak_ssim)/2
print(cropped_source_cloak_dissim)