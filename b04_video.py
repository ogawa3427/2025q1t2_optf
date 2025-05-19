import cv2
import numpy as np
from pathlib import Path

def create_video_from_grids():
    # ディレクトリの設定
    grid_dir = Path("b03_grid_imgs")
    output_dir = Path("b04_videos")
    output_dir.mkdir(exist_ok=True)
    
    # グリッド画像のリストを取得（連番順）
    grid_files = sorted([f for f in grid_dir.glob("*_grid.png")])
    
    if not grid_files:
        print("グリッド画像が見つかりませんでした。")
        return
    
    # 最初の画像から動画の設定を取得
    first_img = cv2.imread(str(grid_files[0]))
    height, width = first_img.shape[:2]
    fps = 30  # フレームレート
    
    # 出力動画の設定
    output_path = output_dir / "grid_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 各グリッド画像を動画に追加
    for grid_file in grid_files:
        img = cv2.imread(str(grid_file))
        if img is None:
            print(f"画像の読み込みに失敗しました: {grid_file}")
            continue
        
        # フレームを動画に追加
        out.write(img)
        print(f"処理完了: {grid_file.name}")
    
    # 動画ファイルを解放
    out.release()
    print(f"動画の生成が完了しました: {output_path}")

if __name__ == "__main__":
    create_video_from_grids() 