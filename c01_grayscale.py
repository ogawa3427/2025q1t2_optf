import cv2
import os
from pathlib import Path

def convert_to_grayscale(input_dir: str, output_dir: str):
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力ディレクトリ内の全PNGファイルを処理
    for img_path in Path(input_dir).glob('*.png'):
        # 画像を読み込み
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: {img_path} の読み込みに失敗しました")
            continue
            
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 出力パスの生成
        output_path = Path(output_dir) / img_path.name
        
        # グレースケール画像の保存
        cv2.imwrite(str(output_path), gray)
        print(f"変換完了: {img_path.name}")

if __name__ == "__main__":
    input_dir = "imgs"
    output_dir = "imgs_grayscale"
    convert_to_grayscale(input_dir, output_dir)
