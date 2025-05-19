import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# 矢印の描画パラメータ
GRID_SIZE = 16  # グリッドのサイズ（ピクセル）
ARROW_SCALE = 500.0  # 矢印のスケール
ARROW_THICKNESS = 2  # 矢印の太さ
NMFACTOR = 1  # フローの大きさに対する色の強さの係数
NMTIMES = 1  # フローの大きさに対する色の強さの指数

# フローの大きさの閾値（色分け用）
FLOW_THRESHOLD_BLUE = 0.05  # 青の最大値
FLOW_THRESHOLD_BLUE_GREEN = 0.1  # 青と緑の中間の最大値
FLOW_THRESHOLD_GREEN = 0.85  # 緑の最大値
FLOW_THRESHOLD_GREEN_RED = 0.87  # 緑と赤の中間の最大値
FLOW_THRESHOLD_RED = 0.9  # 赤の最大値

def plot_flow_magnitude_histogram(flow_magnitudes, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(flow_magnitudes, bins=200, alpha=0.75)
    plt.title('フローベクトルの大きさの分布')
    plt.xlabel('フローの大きさ')
    plt.ylabel('頻度')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def draw_flow_arrows(img, flow, grid_size=GRID_SIZE, scale=ARROW_SCALE, thickness=ARROW_THICKNESS):
    h, w = flow.shape[:2]
    
    # グリッドポイントの生成
    y_coords = np.arange(grid_size//2, h, grid_size)
    x_coords = np.arange(grid_size//2, w, grid_size)
    
    # 全フローの大きさを計算
    flow_magnitudes = []
    for y in y_coords:
        for x in x_coords:
            y1 = max(0, y - grid_size//2)
            y2 = min(h, y + grid_size//2)
            x1 = max(0, x - grid_size//2)
            x2 = min(w, x + grid_size//2)
            mean_flow = np.mean(flow[y1:y2, x1:x2], axis=(0,1))
            flow_magnitude = np.sqrt(mean_flow[0]**2 + mean_flow[1]**2)
            flow_magnitudes.append(flow_magnitude)
    
    # 最頻値を計算
    hist, bins = np.histogram(flow_magnitudes, bins=50)
    mode_index = np.argmax(hist)
    mode_flow_magnitude = (bins[mode_index] + bins[mode_index + 1]) / 2
    
    # グリッド内の平均フローを計算
    for y in y_coords:
        for x in x_coords:
            # グリッド内のフローを取得
            y1 = max(0, y - grid_size//2)
            y2 = min(h, y + grid_size//2)
            x1 = max(0, x - grid_size//2)
            x2 = min(w, x + grid_size//2)
            
            # グリッド内の平均フローを計算
            mean_flow = np.mean(flow[y1:y2, x1:x2], axis=(0,1))
            
            # フローの大きさを計算
            flow_magnitude = np.sqrt(mean_flow[0]**2 + mean_flow[1]**2)
            
            # 矢印の終点を計算（フローの大きさに応じてスケールを調整）
            end_x = int(x + mean_flow[0] * scale * flow_magnitude)
            end_y = int(y + mean_flow[1] * scale * flow_magnitude)
            
            # フローの大きさに応じて色を設定（青→緑→赤）
            # magnitudeを0-1の範囲に正規化（最頻値を中心に）
            normalized_magnitude = min(flow_magnitude / mode_flow_magnitude, 1.0)
            normalized_magnitude = (normalized_magnitude * NMFACTOR) ** NMTIMES
            
            # 5段階の色分け
            if normalized_magnitude < FLOW_THRESHOLD_BLUE:  # 青
                color = (255, 0, 0)  # BGR形式
            elif normalized_magnitude < FLOW_THRESHOLD_BLUE_GREEN:  # 青と緑の中間
                g = int(255 * ((normalized_magnitude - FLOW_THRESHOLD_BLUE) / (FLOW_THRESHOLD_BLUE_GREEN - FLOW_THRESHOLD_BLUE)))
                color = (255, g, 0)  # BGR形式
            elif normalized_magnitude < FLOW_THRESHOLD_GREEN:  # 緑
                color = (0, 255, 0)  # BGR形式
            elif normalized_magnitude < FLOW_THRESHOLD_GREEN_RED:  # 緑と赤の中間
                r = int(255 * ((normalized_magnitude - FLOW_THRESHOLD_GREEN) / (FLOW_THRESHOLD_GREEN_RED - FLOW_THRESHOLD_GREEN)))
                color = (0, 255, r)  # BGR形式
            elif normalized_magnitude < FLOW_THRESHOLD_RED:  # 赤
                color = (0, 0, 255)  # BGR形式
            else:  # 赤以上
                color = (0, 0, 255)  # BGR形式
            
            # 矢印を描画
            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness)

    return img, flow_magnitudes, mode_flow_magnitude

# メイン処理
flow_files = sorted(glob.glob('c04_flow_imgs/flow_*.png'))
os.makedirs('c06_arrow_imgs', exist_ok=True)
os.makedirs('c06_histograms', exist_ok=True)

# 全フローの大きさを収集
all_flow_magnitudes = []
all_normalized_magnitudes = []

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
    
    # 矢印を描画
    result, flow_magnitudes, mode_flow_magnitude = draw_flow_arrows(orig_img.copy(), flow)
    
    # フローの大きさを収集
    normalized_magnitudes = []
    for magnitude in flow_magnitudes:
        normalized = min(magnitude / mode_flow_magnitude, 1.0)
        normalized = (normalized * NMFACTOR) ** NMTIMES
        normalized_magnitudes.append(normalized)
        all_normalized_magnitudes.append(normalized)
    # 個別のヒストグラムを生成
    plot_flow_magnitude_histogram(normalized_magnitudes, f'c06_histograms/hist_{i:04d}.png')
    
    # 保存
    cv2.imwrite(f'c06_arrow_imgs/arrow_{i:04d}.png', result)

# 全フローの大きさのヒストグラムを生成
plot_flow_magnitude_histogram(all_normalized_magnitudes, 'c06_histograms/all_flow_magnitudes.png') 