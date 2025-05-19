import cv2
import os
import argparse

def convert_mp4_to_png(video_path, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    
    # フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"総フレーム数: {total_frames}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # PNGとして保存
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"処理済みフレーム: {frame_count}/{total_frames}")
    
    cap.release()
    print(f"変換完了: {frame_count}フレームを{output_dir}に保存しました")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP4ファイルをPNG連番に変換します")
    parser.add_argument("video_path", help="入力MP4ファイルのパス")
    parser.add_argument("--output_dir", default="output_frames", help="出力ディレクトリのパス")
    
    args = parser.parse_args()
    convert_mp4_to_png(args.video_path, args.output_dir)
