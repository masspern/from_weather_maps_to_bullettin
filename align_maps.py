# ALLINEAMENTO MAPPE CON OPENCV



import cv2
import numpy as np
from PIL import Image
import os

# === CONFIG ===
master_path = '/path/to/template_map.png'
input_dir = '/path/to/all/png/maps/'
output_dir = '/path/to/output/png/maps'

# DIMESIONE DELLA MAPPA MASTER ==> 294, 303

os.makedirs(output_dir, exist_ok=True)

master = cv2.cvtColor(np.array(Image.open(master_path)), cv2.COLOR_RGB2GRAY)
h_master, w_master = master.shape

orb = cv2.ORB_create(5000)

kp_master, des_master = orb.detectAndCompute(master, None)

files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
print(f" Trovate {len(files)} mappe da allineare")

for fname in files:
    path = os.path.join(input_dir, fname)
    img = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2GRAY)
    
    kp_img, des_img = orb.detectAndCompute(img, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_master, des_img)
    
    if len(matches) < 10:
        print(f" Matching debole per {fname}")
        continue

    src_pts = np.float32([kp_master[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    aligned = cv2.warpAffine(cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2BGR), M, (w_master, h_master))

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, aligned)
    print(f" Allineata: {fname}")

print(" Allineamento completato")
