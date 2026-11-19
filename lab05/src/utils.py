# common stuff: loading the Middlebury pair, calibration, disparity with BM / SGBM, error metrics

import csv
import os

import cv2
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")
TMP = os.path.join(RESULTS, "tmp")   # intermediate arrays passed between the scripts, not in the repo

os.makedirs(TMP, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SCALE = 0.5        # the images in data/ are half of the original size
GT_FACTOR = 64     # ground truth is stored as 16-bit png: value = disparity * 64, 0 = unknown


def read_calib(path=os.path.join(DATA, "calib.txt")):
    # Middlebury calib.txt: cam0=[f 0 cx; 0 f cy; 0 0 1], cam1, doffs, baseline (mm), width, ndisp...
    calib = {}
    for line in open(path, encoding="utf-8"):
        key, value = line.strip().split("=")
        if value.startswith("["):
            rows = value.strip("[]").split(";")
            calib[key] = np.array([[float(x) for x in r.split()] for r in rows])
        else:
            calib[key] = float(value)
    return calib


def scaled_calib():
    # the full-size calibration moved to the half-size images: f, cx, cy, doffs and ndisp are
    # multiplied by SCALE, the baseline stays the same
    c = read_calib()
    K0 = c["cam0"].copy()
    K1 = c["cam1"].copy()
    K0[:2] *= SCALE
    K1[:2] *= SCALE
    return {
        "K0": K0, "K1": K1,
        "f": K0[0, 0],
        "B": c["baseline"],             # mm
        "doffs": c["doffs"] * SCALE,    # = cx1 - cx0
        "ndisp": c["ndisp"] * SCALE,
        "size": (int(round(c["width"] * SCALE)), int(round(c["height"] * SCALE))),
    }


def load_pair(kind=""):
    # kind = "" for the perfect pair, "_imperfect" for the imperfectly rectified one
    left = cv2.imread(os.path.join(DATA, f"im0{kind}.png"))
    right = cv2.imread(os.path.join(DATA, f"im1{kind}.png"))
    return left, right


def load_gt(name="disp0_gt.png"):
    # disparity in px of the half-size image, nan where unknown
    raw = cv2.imread(os.path.join(DATA, name), cv2.IMREAD_UNCHANGED).astype(np.float32)
    disp = raw / GT_FACTOR
    disp[raw == 0] = np.nan
    return disp


def make_matcher(method, num_disp, block, uniq, channels=1):
    # the same extra settings for both: speckle filter and left-right check
    if method == "BM":
        m = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
        m.setUniquenessRatio(uniq)
        m.setSpeckleWindowSize(100)
        m.setSpeckleRange(2)
        m.setDisp12MaxDiff(1)
        return m
    # smoothness penalties: the usual 8 * cn * bs^2 and 32 * cn * bs^2
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * channels * block ** 2, P2=32 * channels * block ** 2,
        uniquenessRatio=uniq, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)


def compute_disparity(method, left, right, num_disp, block, uniq, color=False):
    # both work on gray images (SGBM on color overflows its 16-bit costs for blocks >= 13, see
    # disparity.py); the result is fixed point * 16, invalid pixels get (minDisparity - 1) * 16 = -16
    m = make_matcher(method, num_disp, block, uniq, 3 if color else 1)
    if not color:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    raw = m.compute(left, right)
    disp = raw.astype(np.float32) / 16.0
    disp[raw < 0] = np.nan
    return disp


def metrics(disp, gt, mask=None):
    # coverage: share of pixels with a valid value (inside mask if given)
    # bad1 / bad2: share of pixels valid in both with |d - gt| > 1 (2) px
    # correct: share of the ground truth pixels that are valid and within 1 px
    if mask is None:
        mask = np.ones(gt.shape, dtype=bool)
    valid = ~np.isnan(disp) & mask
    known = ~np.isnan(gt) & mask
    both = valid & known
    err = np.abs(disp[both] - gt[both])
    return {
        "coverage": valid.sum() / mask.sum(),
        "bad1": np.mean(err > 1) if len(err) else np.nan,
        "bad2": np.mean(err > 2) if len(err) else np.nan,
        "mae": err.mean() if len(err) else np.nan,
        "correct": np.sum(err <= 1) / known.sum(),
    }


def occlusion_mask(gt0, gt1, tol=1.0):
    # left pixels that have no correspondence in the right image: the right ground truth at
    # x - d does not agree with d (or x - d is outside the image)
    h, w = gt0.shape
    xs = np.arange(w)[None, :] - np.nan_to_num(gt0, nan=0)
    xr = np.round(xs).astype(int)
    inside = (xr >= 0) & (xr < w) & ~np.isnan(gt0)
    d1 = np.full(gt0.shape, np.nan, dtype=np.float32)
    rows = np.repeat(np.arange(h)[:, None], w, axis=1)
    d1[inside] = gt1[rows[inside], xr[inside]]
    agree = np.abs(d1 - gt0) <= tol
    return ~np.isnan(gt0) & ~agree


def depth_from_disparity(disp, cal):
    # Middlebury convention: Z = f * B / (d + doffs), B in mm -> Z in m
    return cal["f"] * cal["B"] / (disp + cal["doffs"]) / 1000.0


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
