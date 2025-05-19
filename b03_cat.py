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
    img_dir = Path("imgs")
    diff_dir = Path("b01_diff_imgs")
    and_dir = Path("b02_and_imgs")
    output_dir = Path("b03_grid_imgs")
    output_dir.mkdir(exist_ok=True)
    
    # 画像ファイルのリストを取得（連番順）
    img_files = sorted([f for f in img_dir.glob("*.png")])
    
    # 各フレームについて処理
    for i in range(1, len(img_files)-1):
        # 必要な画像を読み込み
        original = cv2.imread(str(img_files[i]))
        prev_diff = cv2.imread(str(diff_dir / f"{img_files[i-1].stem}_diff.png"))
        next_diff = cv2.imread(str(diff_dir / f"{img_files[i].stem}_diff.png"))
        and_img = cv2.imread(str(and_dir / f"{img_files[i-1].stem}_and.png"))
        
        if any(img is None for img in [original, prev_diff, next_diff, and_img]):
            print(f"画像の読み込みに失敗しました: {img_files[i]}")
            print(f"  original: {img_files[i]}")
            print(f"  prev_diff: {diff_dir / f'{img_files[i-1].stem}_diff.png'}")
            print(f"  next_diff: {diff_dir / f'{img_files[i].stem}_diff.png'}")
            print(f"  and_img: {and_dir / f'{img_files[i-1].stem}_and.png'}")
            continue
        
        # 2x2グリッドに結合
        grid = create_grid_image([original, prev_diff, next_diff, and_img])
        
        # グリッド画像の保存
        base_name = img_files[i].stem
        cv2.imwrite(str(output_dir / f"{base_name}_grid.png"), grid)
        
        print(f"処理完了: {base_name}")

if __name__ == "__main__":
    main()
