import os
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count()-1))
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MKL_NUM_THREADS"]     = os.environ["OMP_NUM_THREADS"]

import cv2
import numpy as np
import glob
from tqdm import tqdm

WINDOW_SIZE = 15
kernel = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)

def compute_flow_fast(gradient_img):
    # チャネル分割
    It = gradient_img[...,0].astype(np.float32)
    Iy = gradient_img[...,1].astype(np.float32)
    Ix = gradient_img[...,2].astype(np.float32)

    # 要素積
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    # ボックスフィルタでウィンドウ内和を計算
    Sxx = cv2.filter2D(Ix2, -1, kernel)
    Syy = cv2.filter2D(Iy2, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Sxt = -cv2.filter2D(Ixt, -1, kernel)
    Syt = -cv2.filter2D(Iyt, -1, kernel)

    # 判別式
    det = Sxx * Syy - Sxy * Sxy
    eps = 1e-4  # 安定化用

    # 閉形式解
    u = ( Syy * Sxt - Sxy * Syt ) / ( det + eps )
    v = ( Sxx * Syt - Sxy * Sxt ) / ( det + eps )
    return np.stack((u, v), axis=-1)

def visualize_flow(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((*flow.shape[:2],3), dtype=np.uint8)
    hsv[...,0] = ang * 180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# メイン
img_files = sorted(glob.glob('c02_grad_imgs/gradient_*.png'))
os.makedirs('c04_flow_imgs', exist_ok=True)

for i, path in enumerate(tqdm(img_files, desc="光流計算")):
    img = cv2.imread(path)
    flow = compute_flow_fast(img)
    out = visualize_flow(flow)
    cv2.imwrite(f'c04_flow_imgs/flow_{i:04d}.png', out)
