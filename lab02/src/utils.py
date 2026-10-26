# common stuff

import csv
import glob
import os

import cv2
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data", "calib")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BOARD = (9, 6)       # inner corners
SQUARE = 15.5        # mm, measured with a ruler


def image_paths():
    return sorted(glob.glob(os.path.join(DATA, "calib_*.jpg")))


def object_points():
    # (0,0,0), (s,0,0), (2s,0,0), ... row by row, z = 0
    pts = np.zeros((BOARD[0] * BOARD[1], 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE
    return pts


def find_corners(gray):
    # the photos are 1920x2560, on full size findChessboardCorners misses most frames,
    # so try downscaled copies too and scale the corners back
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for scale in [1, 2, 4]:
        small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        ok, corners = cv2.findChessboardCorners(small, BOARD, flags)
        if ok:
            return corners * scale, f"classic /{scale}"
    # last try: the newer detector
    ok, corners = cv2.findChessboardCornersSB(gray, BOARD)
    if ok:
        return corners, "SB"
    return None, "not found"


def refine(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)


def per_frame_errors(objpoints, imgpoints, K, dist, rvecs, tvecs):
    # mean distance between projected and detected corners, for each frame
    errors = []
    for obj, img, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean())
    return np.array(errors)


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
