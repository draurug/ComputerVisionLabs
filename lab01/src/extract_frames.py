# кадри з відео для набору: n штук рівномірно, з сусідніх беру найрізкіший
# python extract_frames.py <відео> 10 rot

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
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

# різкість = дисперсія лапласіана
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

# метадані відео (камера, час початку запису) - через ffprobe
info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", video_path],
                      capture_output=True, text=True)
tags = json.loads(info.stdout)["format"]["tags"]
model = tags.get("com.apple.quicktime.model", "")
record_start = tags.get("com.apple.quicktime.creationdate", "")
print("камера:", model, " початок запису:", record_start)

rows = []
step = len(frames) / n
for k in range(n):
    center = int(step * (k + 0.5))
    # +-10 кадрів
    start = max(0, center - 10)
    end = min(len(frames), center + 11)
    best = start + int(np.argmax(sharpness[start:end]))
    name = f"{prefix}_{k + 1:02d}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), frames[best], [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{name}: кадр {best} ({best / fps:.1f} с), різкість {sharpness[best]:.0f}")
    rows.append([name, os.path.basename(video_path), best, round(best / fps, 2), model, record_start])

with open(os.path.join(out_dir, f"{prefix}_frames.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["файл", "відео", "кадр", "час у відео, с", "камера", "початок запису"])
    writer.writerows(rows)
