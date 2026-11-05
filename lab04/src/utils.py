# common stuff: SIFT matching with the ratio test, reprojection error

import csv
import os

import cv2
import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(LAB, "data")
RESULTS = os.path.join(LAB, "results")
FIG = os.path.join(LAB, "figures")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)


def load(name):
    return cv2.imread(os.path.join(DATA, name))


def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def ratio_matches(desc1, desc2, t=0.7):
    # two nearest neighbours, keep the pair if the first is clearly closer than the second
    good = []
    for m, n in matcher.knnMatch(desc1, desc2, k=2):
        if m.distance < t * n.distance:
            good.append(m)
    return good


def mutual_matches(desc1, desc2, t=0.7):
    # ratio test in both directions, keep a pair only if it is found both ways
    forward = ratio_matches(desc1, desc2, t)
    backward = {m.queryIdx: m.trainIdx for m in ratio_matches(desc2, desc1, t)}
    return [m for m in forward if backward.get(m.trainIdx) == m.queryIdx]


def match_points(kp1, kp2, matches):
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    return src, dst


def reprojection_error(H, src, dst):
    # distance between H * src and dst, per point
    moved = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(moved - dst, axis=1)


def save_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
