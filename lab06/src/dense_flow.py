# Lab 6, part C: dense Farneback flow, HSV picture, time per frame compared with LK

import os
import time

import cv2
import numpy as np

from utils import FB_MAX, FEATURES, FIG, LK, frame_with_label, gray, hstack, read_clip, save_csv, writer

np.random.seed(42)
# one thread for both methods, so the times are comparable and less dependent on the machine load
cv2.setNumThreads(1)

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
SHOW = {"A": 40, "B": 60}   # frame of the clip shown in the figure
V_MAX = 8.0                 # px per frame mapped to full brightness, the same for every frame


def flow_to_bgr(flow):
    # Step 9. direction -> hue, magnitude -> value
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2            # OpenCV hue is 0..180
    hsv[..., 1] = 255
    # a fixed scale instead of per-frame min-max, otherwise noise on a frame without motion looks strong
    hsv[..., 2] = np.clip(mag / V_MAX * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mag


def lk_step(prev, cur, pts):
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
    back, _, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK)
    return nxt, np.linalg.norm((back - pts).reshape(-1, 2), axis=1) <= FB_MAX


rows = []
for name in ["A", "B"]:
    frames = read_clip(name)
    out = writer(f"farneback_{name}.avi", (frames[0].shape[0], frames[0].shape[1] * 2 + 4))
    t_dense, t_lk, moving = [], [], []
    prev = gray(frames[0])
    pts = cv2.goodFeaturesToTrack(prev, **FEATURES)
    for k, frame in enumerate(frames[1:], start=1):
        cur = gray(frame)
        # Step 10. both methods on the same pair of frames, one right after the other
        start = time.perf_counter()
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, **FARNEBACK)
        t_dense.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        lk_step(prev, cur, pts)
        t_lk.append((time.perf_counter() - start) * 1000)

        pic, mag = flow_to_bgr(flow)
        moving.append(np.mean(mag > 1.0))
        out.write(hstack([frame, pic]))
        if k == SHOW[name]:
            cv2.imwrite(os.path.join(FIG, f"farneback_{name}.jpg"),
                        hstack([frame_with_label(frame, f"clip {name}, frame {k}"),
                                frame_with_label(pic, f"flow, V = |v| / {V_MAX:.0f} px")]))
        prev = cur
    out.release()
    h, w = frames[0].shape[:2]
    for label, t in [("Farneback, every pixel", t_dense),
                     (f"LK forward + backward, {len(pts)} points", t_lk)]:
        rows.append([name, f"{w}x{h}", label, round(np.median(t), 2), round(np.percentile(t, 90), 2)])
        print(f"clip {name}: {label}: {np.median(t):.2f} ms per frame (90 %: {np.percentile(t, 90):.2f})")
    print(f"  ratio {np.median(t_dense) / np.median(t_lk):.0f}x, "
          f"pixels with |v| > 1 px: {np.mean(moving) * 100:.2f} %")
    rows.append([name, f"{w}x{h}", "pixels with |v| > 1 px, %", round(np.mean(moving) * 100, 2), ""])

save_csv("timing.csv", ["clip", "size", "method", "median time per frame, ms", "90th percentile, ms"], rows)
