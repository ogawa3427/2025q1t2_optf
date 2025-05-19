import cv2
import numpy as np
from pathlib import Path

def create_and_image(img1, img2):
    """2枚の画像のAND演算を行って返す"""
    # グレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # AND演算
    and_result = cv2.bitwise_and(gray1, gray2)
    
    # 3チャンネルに戻す
    return cv2.cvtColor(and_result, cv2.COLOR_GRAY2BGR)

def main():
    # 差分画像ディレクトリのパス
    diff_dir = Path("b01_diff_imgs")
    # 出力ディレクトリの作成
    output_dir = Path("b02_and_imgs")
    output_dir.mkdir(exist_ok=True)
    
    # 差分画像ファイルのリストを取得（連番順）
    diff_files = sorted([f for f in diff_dir.glob("*_diff.png")])
    
    # 隣り合う画像同士で処理
    for i in range(len(diff_files)-1):
        # 2枚の画像を読み込み
        img1 = cv2.imread(str(diff_files[i]))
        img2 = cv2.imread(str(diff_files[i+1]))
        
        if img1 is None or img2 is None:
            print(f"画像の読み込みに失敗しました: {diff_files[i]}, {diff_files[i+1]}")
            continue
        
        # AND演算画像の作成
        and_result = create_and_image(img1, img2)
        
        # AND演算画像の保存
        base_name = diff_files[i].stem.replace("_diff", "")
        cv2.imwrite(str(output_dir / f"{base_name}_and.png"), and_result)
        
        print(f"処理完了: {base_name}")

if __name__ == "__main__":
    main()
