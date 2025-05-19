import cv2
import numpy as np
from pathlib import Path

def create_grid_image(images, grid_size=(2, 2)):
    """画像を2x2のグリッドに結合する"""
    rows, cols = grid_size
    h, w = images[0].shape[:2]
    
    # グリッド画像の作成
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # 画像を配置
    for idx, img in enumerate(images):
        i, j = divmod(idx, cols)
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img
    
    return grid

def main():
    # 各ディレクトリのパス
    gray_dir = Path("imgs_grayscale")
    slope_dir = Path("c04_flow_imgs")
    and_dir = Path("b02_and_imgs")
    flow_dir = Path("c06_arrow_imgs")
    output_dir = Path("c07_cat_imgs")
    output_dir.mkdir(exist_ok=True)
    
    # 画像ファイルのリストを取得（連番順）
    img_files = sorted([f for f in gray_dir.glob("*.png")])
    
    # 各フレームについて処理
    for img_file in img_files:
        # 必要な画像を読み込み
        gray = cv2.imread(str(gray_dir / img_file.name))
        frame_num = img_file.stem.split('_')[1]  # frame_XXXXXX から XXXXXX を取得
        frame_num_4digit = frame_num.lstrip('0')  # 先頭の0を削除
        if len(frame_num_4digit) < 4:  # 4桁未満の場合
            frame_num_4digit = frame_num_4digit.zfill(4)  # 4桁になるように0埋め
        slope = cv2.imread(str(slope_dir / f"flow_{frame_num_4digit}.png"))
        and_img = cv2.imread(str(and_dir / f"frame_{frame_num}_and.png"))
        flow = cv2.imread(str(flow_dir / f"arrow_{frame_num_4digit}.png"))
        
        if any(img is None for img in [gray, slope, and_img, flow]):
            print(f"画像の読み込みに失敗しました: {img_file}")
            continue
        
        # 2x2グリッドに結合
        grid = create_grid_image([gray, slope, and_img, flow])
        
        # グリッド画像の保存
        base_name = img_file.stem
        cv2.imwrite(str(output_dir / f"{base_name}_cat.png"), grid)
        
        print(f"処理完了: {base_name}")

if __name__ == "__main__":
    main()
