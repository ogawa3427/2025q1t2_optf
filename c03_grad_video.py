import cv2
import os
import glob
import numpy as np

def create_video_from_images(image_dir, output_path, fps=30):
    # 画像ファイルのパスを取得してソート
    image_files = sorted(glob.glob(os.path.join(image_dir, 'gradient_*.png')))
    
    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        return
    
    # 最初の画像を読み込んでサイズを取得
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 各画像を動画に追加
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is not None:
            out.write(img)
            print(f"処理中: {os.path.basename(image_file)}")
    
    # リソースの解放
    out.release()
    print(f"動画の作成が完了しました: {output_path}")

if __name__ == "__main__":
    # 入力ディレクトリと出力ファイルのパスを設定
    input_dir = "c02_grad_imgs"
    output_file = "output_gradient.mp4"
    
    # 動画の作成
    create_video_from_images(input_dir, output_file)
