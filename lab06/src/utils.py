# common stuff: paths, the two clips cut from vtest.avi, csv

import csv
import os

import cv2
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")
TMP = os.path.join(RESULTS, "tmp")

for d in (RESULTS, FIG, TMP):
    os.makedirs(d, exist_ok=True)

VIDEO = os.path.join(DATA, "vtest.avi")
FPS = 10

# Step 3-5, shared by the sparse flow and the timing in dense_flow.py
# maxLevel = 2 means three pyramid levels: the image itself and two halvings
FEATURES = dict(maxCorners=300, qualityLevel=0.01, minDistance=10)
LK = dict(winSize=(21, 21), maxLevel=2,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FB_MAX = 1.0      # px, forward-backward error limit

# frame ranges (first, last, inclusive) and ROI (x, y, w, h) in the original video
CLIPS = {
    # one woman walking on the lawn; only the legs of a passer-by on the road touch the top right
    # corner in frames 683-702
    "A": {"first": 605, "last": 709, "roi": (0, 270, 440, 306)},
    # several people, the woman in red goes past a man and behind the lamp post sign
    "B": {"first": 420, "last": 619, "roi": (0, 0, 768, 576)},
}


def read_clip(name):
    clip = CLIPS[name]
    x, y, w, h = clip["roi"]
    # read from the start instead of seeking: seeking in a compressed avi can land on the wrong frame
    video = cv2.VideoCapture(VIDEO)
    frames = []
    index = 0
    while index <= clip["last"]:
        ok, frame = video.read()
        if not ok:
            break
        if index >= clip["first"]:
            frames.append(frame[y:y + h, x:x + w].copy())
        index += 1
    video.release()
    return frames


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def writer(name, shape, fps=FPS):
    h, w = shape[:2]
    return cv2.VideoWriter(os.path.join(TMP, name), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_with_label(frame, text):
    out = frame.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def hstack(images):
    # side by side with a 4 px white gap between the images
    h = max(im.shape[0] for im in images)
    padded = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0 if i == len(images) - 1 else 4,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for i, im in enumerate(images)]
    return np.hstack(padded)
