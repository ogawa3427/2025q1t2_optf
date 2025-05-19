import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# 矢印の描画パラメータ
GRID_SIZE = 16  # グリッドのサイズ（ピクセル）
ARROW_SCALE = 800.0  # 矢印のスケール
ARROW_THICKNESS = 2  # 矢印の太さ
MIN_SCALE = 400.0  # 最小スケール
MAX_SCALE = 1200.0  # 最大スケール

def draw_flow_arrows_dynamic(img, flow, grid_size=GRID_SIZE, scale=ARROW_SCALE, thickness=ARROW_THICKNESS):
    h, w = flow.shape[:2]
    # グリッド生成
    ys = np.arange(grid_size//2, h, grid_size)
    xs = np.arange(grid_size//2, w, grid_size)
    # 各セルの平均フロー大きさを収集
    all_mags = []
    for y in ys:
        for x in xs:
            y1, y2 = max(0,y-grid_size//2), min(h,y+grid_size//2)
            x1, x2 = max(0,x-grid_size//2), min(w,x+grid_size//2)
            mv = np.mean(flow[y1:y2, x1:x2], axis=(0,1))
            all_mags.append(np.linalg.norm(mv))
    max_m = max(all_mags) or 1.0

    for y in ys:
        for x in xs:
            y1, y2 = max(0,y-grid_size//2), min(h,y+grid_size//2)
            x1, x2 = max(0,x-grid_size//2), min(w,x+grid_size//2)
            mv = np.mean(flow[y1:y2, x1:x2], axis=(0,1))
            m = np.linalg.norm(mv)
            norm = m / max_m  # 0–1
            # Hue を 120(青)→0(赤) に線形マッピング
            hue = int((1.0 - norm) * 120)
            hsv_pix = np.uint8([[[hue, 255, 255]]])
            col = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0,0].tolist()
            ex, ey = int(x+mv[0]*scale*m), int(y+mv[1]*scale*m)
            cv2.arrowedLine(img, (x,y), (ex,ey), col, thickness)
    return img

# メイン処理
flow_files = sorted(glob.glob('c04_flow_imgs/flow_*.png'))
os.makedirs('c06_arrow_imgs', exist_ok=True)

for i, path in enumerate(tqdm(flow_files, desc="矢印描画")):
    # フロー画像を読み込み
    flow_img = cv2.imread(path)
    
    # HSVからフローを再構築
    hsv = cv2.cvtColor(flow_img, cv2.COLOR_BGR2HSV)
    mag = hsv[...,2].astype(np.float32) / 255.0
    ang = hsv[...,0].astype(np.float32) * 2 * np.pi / 180.0
    
    # 極座標から直交座標に変換
    flow = np.stack((
        mag * np.cos(ang),
        mag * np.sin(ang)
    ), axis=-1)
    
    # 元画像を読み込み（グレースケール）
    orig_img = cv2.imread(f'imgs_grayscale/frame_{i:06d}.png', cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    
    # 動的な矢印を描画
    result = draw_flow_arrows_dynamic(orig_img.copy(), flow)
    
    # 保存
    cv2.imwrite(f'c06_arrow_imgs/arrow_{i:04d}.png', result) 