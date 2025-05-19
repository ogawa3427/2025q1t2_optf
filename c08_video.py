import cv2
import numpy as np
import os
from pathlib import Path
import glob

def create_video_from_images(image_dir, output_path, fps=30, pattern="*.png"):
    """
    指定したディレクトリ内の画像から動画を作成する
    
    Args:
        image_dir: 画像ファイルのディレクトリパス
        output_path: 出力動画のパス
        fps: フレームレート（デフォルト30）
        pattern: 画像ファイルの検索パターン（デフォルト*.png）
    """
    # 画像ファイルのパスを取得してソート
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    
    image_files = sorted(list(image_dir.glob(pattern)))
    
    if not image_files:
        print(f"画像ファイルが見つかりませんでした。パターン: {pattern}")
        return
    
    # 最初の画像を読み込んでサイズを取得
    first_image = cv2.imread(str(image_files[0]))
    height, width, layers = first_image.shape
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(output_path)
    if output_dir:  # ディレクトリパスが存在する場合のみ作成
        os.makedirs(output_dir, exist_ok=True)
    
    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 各画像を動画に追加
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        if img is not None:
            out.write(img)
            print(f"処理中: {image_file.name}")
        else:
            print(f"画像の読み込みに失敗: {image_file}")
    
    # リソースの解放
    out.release()
    print(f"動画の作成が完了しました: {output_path}")

def extract_frames_from_video(video_path, output_dir, prefix="frame", ext=".png", step=1):
    """
    動画からフレームを抽出して画像として保存する
    
    Args:
        video_path: 動画ファイルのパス
        output_dir: 出力先ディレクトリ
        prefix: 出力ファイル名の接頭辞（デフォルト"frame"）
        ext: 出力ファイルの拡張子（デフォルト".png"）
        step: 抽出するフレームの間隔（デフォルト1=全フレーム）
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画ファイルを開けませんでした: {video_path}")
        return
    
    frame_count = 0
    save_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ステップごとにフレームを保存
        if frame_count % step == 0:
            output_path = os.path.join(output_dir, f"{prefix}_{save_count:04d}{ext}")
            cv2.imwrite(output_path, frame)
            save_count += 1
            print(f"フレーム保存: {output_path}")
        
        frame_count += 1
    
    # リソースの解放
    cap.release()
    print(f"フレーム抽出が完了しました。合計{save_count}枚のフレームを保存しました。")

if __name__ == "__main__":
    # c07_cat_imgsディレクトリの画像から動画を作成
    input_dir = "c07_cat_imgs"
    output_file = "c08_cat_video.mp4"
    
    print(f"画像ディレクトリ: {input_dir}")
    print(f"出力ファイル: {output_file}")
    
    create_video_from_images(input_dir, output_file)
    
    # 動画からフレームを抽出
    # extract_frames_from_video("video.mp4", "extracted_frames")
    
    print("使用例:")
    print("1. 画像から動画を作成: create_video_from_images('入力ディレクトリ', '出力ファイル名.mp4')")
    print("2. 動画からフレームを抽出: extract_frames_from_video('入力動画.mp4', '出力ディレクトリ')")
