# extract_superpoint.py

# Run from build 
# python ../modern_features/extract_superpoint.py ../modern_features/superpoint_v1.pth ../datasets/images_1 ../datasets/3dp_cam.yml features/ 1.1

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1,  c1, 3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, 3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, 3, stride=1, padding=1)
        self.convPa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65,  1, stride=1, padding=0)
        self.convDa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = desc / (torch.norm(desc, p=2, dim=1, keepdim=True) + 1e-8)
        return scores, desc


def extract(model, img_bgr, intrinsics, dist_coeffs, new_intrinsics,
            nms_radius=3, score_thresh=0.005, max_kp=3000):

    und = cv2.undistort(img_bgr, intrinsics, dist_coeffs, None, new_intrinsics)
    H = (und.shape[0] // 8) * 8
    W = (und.shape[1] // 8) * 8
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0

    inp = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        scores_raw, desc_raw = model(inp)

    Hc, Wc = H // 8, W // 8
    scores = scores_raw.squeeze(0)
    scores = torch.nn.functional.softmax(scores, dim=0)[:64]
    scores = scores.permute(1, 2, 0).reshape(Hc, Wc, 8, 8)
    scores = scores.permute(0, 2, 1, 3).reshape(H, W).numpy()

    # NMS
    import torch.nn.functional as F
    ht = torch.from_numpy(scores).unsqueeze(0).unsqueeze(0)
    local_max = F.max_pool2d(ht, kernel_size=nms_radius*2+1,
                              stride=1, padding=nms_radius)
    mask = ((ht == local_max) & (ht > score_thresh)).squeeze().numpy()

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 256), np.float32)

    vals = scores[ys, xs]
    if len(vals) > max_kp:
        idx = np.argsort(vals)[::-1][:max_kp]
        xs, ys = xs[idx], ys[idx]

    sx = und.shape[1] / W
    sy = und.shape[0] / H
    kpts = np.stack([xs * sx, ys * sy], axis=1).astype(np.float32)

    desc_np = desc_raw.squeeze(0).numpy()  # (256, Hc, Wc)
    fpx = np.clip(xs / 8.0, 0, Wc - 1)
    fpy = np.clip(ys / 8.0, 0, Hc - 1)
    x0 = fpx.astype(int); x1 = np.clip(x0+1, 0, Wc-1)
    y0 = fpy.astype(int); y1 = np.clip(y0+1, 0, Hc-1)
    wx = fpx - x0; wy = fpy - y0

    descs = (((1-wy)*(1-wx)) * desc_np[:, y0, x0] +
             ((1-wy)*(  wx)) * desc_np[:, y0, x1] +
             ((  wy)*(1-wx)) * desc_np[:, y1, x0] +
             ((  wy)*(  wx)) * desc_np[:, y1, x1]).T  # (N, 256)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norms + 1e-8)

    return kpts, descs


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python extract_superpoint.py <weights.pth> <image_dir> <calib.yml> <output_dir> [focal_scale]")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_dir    = sys.argv[2]
    calib_path   = sys.argv[3]
    output_dir   = sys.argv[4]
    focal_scale  = float(sys.argv[5]) if len(sys.argv) > 5 else 1.1

    os.makedirs(output_dir, exist_ok=True)

    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    K    = fs.getNode('K').mat()
    dist = fs.getNode('D').mat()
    fs.release()

    new_K = K.copy()
    new_K[0, 0] *= focal_scale
    new_K[1, 1] *= focal_scale

    model = SuperPointNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print("SuperPoint loaded.")

    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith(('.jpg','.png','.jpeg'))])

    for idx, img_name in enumerate(images):
        img = cv2.imread(os.path.join(image_dir, img_name))
        kpts, descs = extract(model, img, K, dist, new_K)

        # One .txt file per image: first line = N
        # Then N lines of: x y d0 d1 ... d255
        out_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(out_path, 'w') as f:
            f.write(f"{len(kpts)}\n")
            for k, d in zip(kpts, descs):
                line = f"{k[0]:.4f} {k[1]:.4f} " + " ".join(f"{v:.6f}" for v in d)
                f.write(line + "\n")

        print(f"  [{idx}] {img_name}: {len(kpts)} keypoints -> {out_path}")

    print(f"\nDone. Features saved to {output_dir}")