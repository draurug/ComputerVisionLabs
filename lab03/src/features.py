# Lab 3, part C: SIFT and ORB, keypoints and descriptors

import os
import time

import cv2
import numpy as np

from utils import FIG, N_POINTS, image_paths, save_csv

sift = cv2.SIFT_create(nfeatures=N_POINTS)
orb = cv2.ORB_create(nfeatures=N_POINTS)

rows = []
for path in image_paths():
    name = os.path.basename(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for label, det in [("SIFT", sift), ("ORB", orb)]:
        # time: median of 5 runs
        times = []
        for _ in range(5):
            start = time.perf_counter()
            kps, desc = det.detectAndCompute(gray, None)
            times.append((time.perf_counter() - start) * 1000)
        print(f"{name} {label}: {len(kps)} points, descriptor {desc.shape[1]} x {desc.dtype}, "
              f"{np.median(times):.1f} ms")
        rows.append([name, label, len(kps), desc.shape[1], str(desc.dtype), round(np.median(times), 1)])

        if name == os.path.basename(image_paths()[0]):
            # Step 13. circle radius = scale, line = orientation
            pic = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(FIG, f"keypoints_{label.lower()}.jpg"), pic)

save_csv("sift_orb.csv", ["file", "detector", "points", "descriptor size", "dtype", "time, ms"], rows)
