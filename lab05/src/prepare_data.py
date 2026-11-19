# Lab 5, part A (variant 3): Middlebury 2014 "Piano" -> half-size images and ground truth in data/
#
#   python prepare_data.py <folder>
#
# <folder> has Piano-perfect.zip and Piano-imperfect.zip (downloaded if missing, ~165 MB)

import os
import shutil
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

from utils import DATA, GT_FACTOR, SCALE

URL = "https://vision.middlebury.edu/stereo/data/scenes2014/zip/{}.zip"
SCENE = "Piano"


def read_pfm(data):
    # PFM: "Pf" (1 channel), "width height", scale (< 0 = little endian), rows from bottom to top
    lines = data.split(b"\n", 3)
    w, h = map(int, lines[1].split())
    endian = "<" if float(lines[2]) < 0 else ">"
    disp = np.frombuffer(lines[3], dtype=endian + "f4", count=w * h).reshape(h, w)
    return np.flipud(disp).astype(np.float32)


def half(image):
    return cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def half_disp(disp):
    # nearest neighbour, so that no value is averaged across an object border; disparity * SCALE
    small = cv2.resize(disp, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST) * SCALE
    small[~np.isfinite(small)] = 0
    return np.round(small * GT_FACTOR).astype(np.uint16)


folder = sys.argv[1]
os.makedirs(folder, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
for version, suffix in [("perfect", ""), ("imperfect", "_imperfect")]:
    path = os.path.join(folder, f"{SCENE}-{version}.zip")
    if not os.path.exists(path):
        print("downloading", URL.format(f"{SCENE}-{version}"))
        with urllib.request.urlopen(URL.format(f"{SCENE}-{version}")) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    with zipfile.ZipFile(path) as z:
        prefix = f"{SCENE}-{version}/"
        for name in ["im0.png", "im1.png"]:
            buf = np.frombuffer(z.read(prefix + name), np.uint8)
            cv2.imwrite(os.path.join(DATA, name.replace(".png", suffix + ".png")),
                        half(cv2.imdecode(buf, cv2.IMREAD_COLOR)))
        if version == "perfect":
            open(os.path.join(DATA, "calib.txt"), "wb").write(z.read(prefix + "calib.txt"))
            for name in ["disp0", "disp1"]:
                cv2.imwrite(os.path.join(DATA, f"{name}_gt.png"), half_disp(read_pfm(z.read(prefix + name + ".pfm"))))
        else:
            open(os.path.join(DATA, "calib_imperfect.txt"), "wb").write(z.read(prefix + "calib.txt"))
    print(version, "done")
