import os
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count()-1))
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MKL_NUM_THREADS"]     = os.environ["OMP_NUM_THREADS"]

import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'

# メイン
img_files = sorted(glob.glob('c04_flow_imgs/flow_*.png'))
os.makedirs('d01_histograms', exist_ok=True)

for i, path in enumerate(tqdm(img_files, desc="光流ヒストグラム生成")):
    # HSV画像から光流を復元
    hsv = cv2.imread(path)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    
    # 角度と大きさから光流を復元
    mag = hsv[..., 2].astype(np.float32) / 255.0
    ang = hsv[..., 0].astype(np.float32) * 2 * np.pi / 180.0
    
    # 極座標から直交座標に変換
    u = mag * np.cos(ang)
    v = mag * np.sin(ang)
    
    flow = np.stack((u, v), axis=-1)
    
    # 各フレームのヒストグラムを作成
    plt.figure(figsize=(20, 15))
    
    # X成分のヒストグラム
    plt.subplot(3, 1, 1)
    plt.hist(flow[..., 0].flatten(), bins=200, alpha=0.7, color='blue')
    plt.title(f'フレーム {i+1} のX成分ヒストグラム')
    plt.xlabel('X成分の値')
    plt.ylabel('頻度 (対数スケール)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Y成分のヒストグラム
    plt.subplot(3, 1, 2)
    plt.hist(flow[..., 1].flatten(), bins=200, alpha=0.7, color='red')
    plt.title(f'フレーム {i+1} のY成分ヒストグラム')
    plt.xlabel('Y成分の値')
    plt.ylabel('頻度 (対数スケール)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # フローの大きさのヒストグラム
    plt.subplot(3, 1, 3)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    plt.hist(magnitude.flatten(), bins=200, alpha=0.7, color='green')
    plt.title(f'フレーム {i+1} のフロー大きさヒストグラム')
    plt.xlabel('フローの大きさ')
    plt.ylabel('頻度 (対数スケール)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'd01_histograms/flow_histogram_frame_{i+1:04d}.png', dpi=300)
    plt.close()