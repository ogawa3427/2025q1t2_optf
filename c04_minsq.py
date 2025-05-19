import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# スレッド制御（お好みで）
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count()-1))
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MKL_NUM_THREADS"]     = os.environ["OMP_NUM_THREADS"]

WINDOW_SIZE = 15
PAD = WINDOW_SIZE // 2
EPS = 1e-4

def compute_flow_bruteforce(grad_img):
    """
    grad_img: H x W x 3 の np.uint8 画像
      channel 0 = It, 1 = Iy, 2 = Ix
    """
    H, W, _ = grad_img.shape
    # float32 に変換してチャンネル分解
    It = grad_img[...,0].astype(np.float32) * -1.0   # -∂I/∂t
    Iy = grad_img[...,1].astype(np.float32)          # ∂I/∂y
    Ix = grad_img[...,2].astype(np.float32)          # ∂I/∂x

    # 境界反射パディング
    It_p = np.pad(It,  PAD, mode='reflect')
    Iy_p = np.pad(Iy,  PAD, mode='reflect')
    Ix_p = np.pad(Ix,  PAD, mode='reflect')

    flow = np.zeros((H, W, 2), dtype=np.float32)

    # 各ピクセルごとにウィンドウ内を総当り計算
    for y in range(H):
        for x in range(W):
            Sxx = Syy = Sxy = Sxt = Syt = 0.0
            # ウィンドウ内ループ
            for dy in range(-PAD, PAD+1):
                for dx in range(-PAD, PAD+1):
                    ix = x + dx + PAD
                    iy = y + dy + PAD
                    ix_val = Ix_p[iy, ix]
                    iy_val = Iy_p[iy, ix]
                    it_val = It_p[iy, ix]

                    Sxx  += ix_val * ix_val
                    Syy  += iy_val * iy_val
                    Sxy  += ix_val * iy_val
                    Sxt  += ix_val * it_val
                    Syt  += iy_val * it_val
                    print(f"progress: {x}/{W}, {y}/{H}")

            # 正規方程式 A (2x2) と b (2x1)
            det = Sxx * Syy - Sxy * Sxy
            # 小さすぎるときは eps 加算
            if abs(det) < EPS:
                flow[y, x, :] = 0.0
            else:
                u = ( Syy * Sxt - Sxy * Syt ) / (det + EPS)
                v = ( Sxx * Syt - Sxy * Sxt ) / (det + EPS)
                flow[y, x, 0] = u
                flow[y, x, 1] = v

    return flow

def visualize_flow(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((*flow.shape[:2],3), dtype=np.uint8)
    hsv[...,0] = ang * 180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    img_files = sorted(glob.glob('c02_grad_imgs/gradient_*.png'))
    os.makedirs('c04_flow_bruteforce', exist_ok=True)

    for i, path in enumerate(tqdm(img_files, desc="泥臭い光流計算")):
        img = cv2.imread(path)
        flow = compute_flow_bruteforce(img)
        out  = visualize_flow(flow)
        cv2.imwrite(f'c04_flow_bruteforce/flow_{i:04d}.png', out)
