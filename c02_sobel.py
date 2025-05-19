import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# 画像ファイルのパスを取得
img_dir = 'imgs_grayscale'
img_files = sorted(glob.glob(os.path.join(img_dir, 'frame_*.png')))

# 出力ディレクトリの作成
output_dir = 'c02_grad_imgs'
os.makedirs(output_dir, exist_ok=True)

# 最初のフレームを読み込み
prev_gray = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)

for i in tqdm(range(1, len(img_files)), desc="勾配計算中"):
    # 現在のフレームを読み込み
    next_gray = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
    
    # 空間勾配（Sobel フィルタ）
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 時間勾配
    It = next_gray.astype(np.float64) - prev_gray.astype(np.float64)

    # 勾配情報をRGB画像に変換
    # 各勾配の値を0-255の範囲に正規化
    Ix_norm = cv2.normalize(Ix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Iy_norm = cv2.normalize(Iy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    It_norm = cv2.normalize(It, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # RGB画像の作成（Ix→R, Iy→G, It→B）
    gradient_rgb = cv2.merge([Ix_norm, Iy_norm, It_norm])

    # 結果を保存
    output_path = os.path.join(output_dir, f'gradient_{i:04d}.png')
    cv2.imwrite(output_path, gradient_rgb)

    # フレームの更新
    prev_gray = next_gray.copy()

print(f'勾配情報の画像を {output_dir} に保存しました。')
