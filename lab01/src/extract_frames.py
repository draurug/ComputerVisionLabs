# frames from a video for the dataset: n frames evenly, the sharpest one among neighbours
# python extract_frames.py <video> 10 rot

import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from utils import DATA

video_path = sys.argv[1]
n = int(sys.argv[2])
prefix = sys.argv[3]

out_dir = os.path.join(DATA, "dataset")
os.makedirs(out_dir, exist_ok=True)

video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)

# sharpness = variance of the laplacian
frames = []
sharpness = []
while True:
    ok, frame = video.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
    sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
video.release()

# video metadata (camera, recording start) - with ffprobe
info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", video_path],
                      capture_output=True, text=True)
tags = json.loads(info.stdout)["format"]["tags"]
model = tags.get("com.apple.quicktime.model", "")
record_start = tags.get("com.apple.quicktime.creationdate", "")
print("camera:", model, " recording start:", record_start)

rows = []
step = len(frames) / n
for k in range(n):
    center = int(step * (k + 0.5))
    # +-10 frames
    start = max(0, center - 10)
    end = min(len(frames), center + 11)
    best = start + int(np.argmax(sharpness[start:end]))
    name = f"{prefix}_{k + 1:02d}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), frames[best], [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{name}: frame {best} ({best / fps:.1f} s), sharpness {sharpness[best]:.0f}")
    rows.append([name, os.path.basename(video_path), best, round(best / fps, 2), model, record_start])

with open(os.path.join(out_dir, f"{prefix}_frames.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "video", "frame", "time_s", "camera", "record_start"])
    writer.writerows(rows)
