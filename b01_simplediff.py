import cv2
import numpy as np
import os
from pathlib import Path

def create_diff_image(img1, img2):
    """2枚の画像の差分を計算して返す"""
    diff = cv2.absdiff(img1, img2)
    return diff

def main():
    # 画像ディレクトリのパス
    img_dir = Path("imgs")
    # 出力ディレクトリの作成
    output_dir = Path("b01_diff_imgs")
    output_dir.mkdir(exist_ok=True)
    
    # 画像ファイルのリストを取得（連番順）
    img_files = sorted([f for f in img_dir.glob("*.png")])
    
    # 2枚ずつ処理
    for i in range(len(img_files)-1):
        # 2枚の画像を読み込み
        img1 = cv2.imread(str(img_files[i]))
        img2 = cv2.imread(str(img_files[i+1]))
        
        if img1 is None or img2 is None:
            print(f"画像の読み込みに失敗しました: {img_files[i]}, {img_files[i+1]}")
            continue
        
        # 差分画像の作成
        diff = create_diff_image(img1, img2)
        
        # 差分画像の保存
        base_name = img_files[i].stem
        cv2.imwrite(str(output_dir / f"{base_name}_diff.png"), diff)
        
        print(f"処理完了: {base_name}")

if __name__ == "__main__":
    main()
