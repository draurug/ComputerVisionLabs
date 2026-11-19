# Lab 5, part B: Q from stereoRectify, line grid check, vertical misalignment |yL - yR|

import os

import cv2
import numpy as np

from utils import DATA, FIG, TMP, load_pair, read_calib, save_csv, scaled_calib

np.random.seed(42)

cal = scaled_calib()
left, right = load_pair()
w, h = cal["size"]
print(f"size {w}x{h}, f = {cal['f']:.2f} px, B = {cal['B']:.2f} mm, doffs = {cal['doffs']:.3f} px, "
      f"ndisp = {cal['ndisp']:.0f}")

# Step 4. The Middlebury pair is already rectified, so there is no stereoCalibrate: R = I and
# T = (-B, 0, 0) come from the dataset calibration. stereoRectify gives Q, then the maps and remap.
R = np.eye(3)
T = np.array([[-cal["B"]], [0], [0]], dtype=np.float64)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cal["K0"], None, cal["K1"], None, (w, h), R, T,
                                            flags=0, alpha=-1)
print("R1 = I:", np.allclose(R1, np.eye(3)), " R2 = I:", np.allclose(R2, np.eye(3)))
print("Q =\n", np.round(Q, 5))
# Z = f / (Q[3,2] * d + Q[3,3])  ->  f = Q[2,3], B = 1 / Q[3,2], doffs = Q[3,3] / Q[3,2]
f_q = Q[2, 3]
B_q = 1 / Q[3, 2]
doffs_q = Q[3, 3] / Q[3, 2]
print(f"from Q: f = {f_q:.2f}, B = {B_q:.2f} mm, doffs = {doffs_q:.3f}")

map_l = cv2.initUndistortRectifyMap(cal["K0"], None, R1, P1, (w, h), cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(cal["K1"], None, R2, P2, (w, h), cv2.CV_32FC1)
rect_l = cv2.remap(left, *map_l, cv2.INTER_LINEAR)
rect_r = cv2.remap(right, *map_r, cv2.INTER_LINEAR)
# how far the maps move the pixels: with R = I and the same K the remap is the identity
grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
shift_l = np.hypot(map_l[0] - grid_x, map_l[1] - grid_y)
shift_r = np.hypot(map_r[0] - grid_x, map_r[1] - grid_y)
print(f"remap shift: left max {shift_l.max():.3f} px, right max {shift_r.max():.3f} px")
np.save(os.path.join(TMP, "Q.npy"), Q)

save_csv("rectify_q.csv", ["quantity", "dataset calib", "from Q"], [
    ["f, px", round(cal["f"], 3), round(f_q, 3)],
    ["B, mm", round(cal["B"], 3), round(B_q, 3)],
    ["doffs, px", round(cal["doffs"], 3), round(doffs_q, 3)],
    ["max remap shift left, px", "", round(float(shift_l.max()), 4)],
    ["max remap shift right, px", "", round(float(shift_r.max()), 4)],
])


# Step 5. Horizontal lines every 25 px over both images
def lines(a, b, step=25):
    pic = np.hstack([a, b])
    for y in range(0, pic.shape[0], step):
        cv2.line(pic, (0, y), (pic.shape[1], y), (0, 255, 0), 1)
    return pic


cv2.imwrite(os.path.join(FIG, "rectified_lines.jpg"), lines(rect_l, rect_r))

# Step 6. Vertical misalignment. Instead of marking by hand: 15 corners, the strongest
# Shi-Tomasi corner in each cell of a 5 x 3 grid (refined with cornerSubPix) - the kind of points one
# marks by hand - and their partners in the right image found by normalized correlation of a 21 x 21
# patch over the whole disparity range and +-5 rows, with a parabola through the peak.
# The same is also measured over all SIFT matches (ratio test 0.7, mutual, RANSAC on F).
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)


def matches(a, b, t=0.7):
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    fwd = {m.queryIdx: (m.trainIdx, m.distance / n.distance)
           for m, n in matcher.knnMatch(da, db, k=2) if m.distance < t * n.distance}
    bwd = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(db, da, k=2) if m.distance < t * n.distance}
    pairs = [(i, j, r) for i, (j, r) in fwd.items() if bwd.get(j) == i]
    pa = np.float32([ka[i].pt for i, _, _ in pairs])
    pb = np.float32([kb[j].pt for _, j, _ in pairs])
    ratio = np.float32([r for _, _, r in pairs])
    # wrong matches are removed with the epipolar geometry (RANSAC on F, 1 px), not with dy itself
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
    keep = mask.ravel().astype(bool)
    return pa[keep], pb[keep], ratio[keep]


def ncc_search(ga, gb, x, y, max_disp, r, s):
    # partner of the pixel (x, y) of ga in gb: x +- max_disp, y +- s, subpixel peak
    patch = ga[y - r:y + r + 1, x - r:x + r + 1]
    area = gb[y - r - s:y + r + s + 1, x - max_disp - r:x + max_disp + r + 1]
    score = cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED)
    py, px = np.unravel_index(np.argmax(score), score.shape)
    if not (0 < px < score.shape[1] - 1 and 0 < py < score.shape[0] - 1):
        return None

    def sub(m, c, p):
        return 0.5 * (m - p) / (m - 2 * c + p)

    xb = x - max_disp + px + sub(score[py, px - 1], score[py, px], score[py, px + 1])
    yb = y - s + py + sub(score[py - 1, px], score[py, px], score[py + 1, px])
    return xb, yb, score[py, px]


def corner_pairs(a, b, max_disp, nx=5, ny=3, r=10, s=5, min_score=0.9):
    # in each cell: up to 5 strongest corners, the first one whose match is sure is taken -
    # correlation >= 0.9 and the search back from the right image returns to it within 1 px
    # (this drops corners on repeated texture and at depth edges, as one would do by hand)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h, w = ga.shape
    pad = r + s + max_disp + 2
    pairs = []
    for cy in range(ny):
        for cx in range(nx):
            mask = np.zeros_like(ga)
            x0, x1 = int(cx * w / nx), int((cx + 1) * w / nx)
            y0, y1 = int(cy * h / ny), int((cy + 1) * h / ny)
            mask[max(y0, r + s + 2):min(y1, h - r - s - 2), max(x0, pad):min(x1, w - pad)] = 255
            corners = cv2.goodFeaturesToTrack(ga, 5, 0.01, 10, mask=mask)
            if corners is None:
                continue
            corners = cv2.cornerSubPix(ga, corners.astype(np.float32), (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            for xa, ya in corners.reshape(-1, 2):
                xi, yi = int(round(xa)), int(round(ya))
                fwd = ncc_search(ga, gb, xi, yi, max_disp, r, s)
                if fwd is None or fwd[2] < min_score:
                    continue
                back = ncc_search(gb, ga, int(round(fwd[0])), int(round(fwd[1])), max_disp, r, s)
                if back is None or abs(back[0] - xi) > 1 or abs(back[1] - yi) > 1:
                    continue
                # the patch was cut at the rounded corner, so the subpixel part of the corner is kept
                pairs.append(((xa, ya), (fwd[0] + xa - xi, fwd[1] + ya - yi), fwd[2]))
                break
    return pairs


def misalignment(a, b, label):
    pa, pb, _ = matches(a, b)
    dy = np.abs(pa[:, 1] - pb[:, 1])
    picked = corner_pairs(a, b, int(cal["ndisp"]))
    dy_p = np.array([abs(p[1] - q[1]) for p, q, _ in picked])
    print(f"{label}: {len(pa)} matches, |dy| mean {dy.mean():.3f}, median {np.median(dy):.3f}, "
          f"95% {np.percentile(dy, 95):.3f} px; {len(picked)} corners: mean {dy_p.mean():.3f}, "
          f"median {np.median(dy_p):.3f}, max {dy_p.max():.3f} px, min correlation {min(c for _, _, c in picked):.3f}")
    row = [label, len(pa), round(dy.mean(), 3), round(np.median(dy), 3), round(np.percentile(dy, 95), 3),
           len(picked), round(dy_p.mean(), 3), round(np.median(dy_p), 3), round(dy_p.max(), 3)]
    points = [[label, i + 1, *np.round(p, 2), *np.round(q, 2), round(abs(p[1] - q[1]), 3), round(c, 3)]
              for i, (p, q, c) in enumerate(picked)]
    return row, points, pa, pb


rows, points = [], []
r, p, _, _ = misalignment(rect_l, rect_r, "perfect")
rows.append(r)
points += p

# The imperfect version of the same scene: the dataset rectification has a small error.
# Fix it without calibration: F from the matches -> stereoRectifyUncalibrated -> two homographies.
left_i, right_i = load_pair("_imperfect")
r, p, pa_i, pb_i = misalignment(left_i, right_i, "imperfect")
rows.append(r)
points += p
F, _ = cv2.findFundamentalMat(pa_i, pb_i, cv2.FM_8POINT)
ok, H1, H2 = cv2.stereoRectifyUncalibrated(pa_i, pb_i, F, (w, h))
fixed_l = cv2.warpPerspective(left_i, H1, (w, h))
fixed_r = cv2.warpPerspective(right_i, H2, (w, h))
r, p, _, _ = misalignment(fixed_l, fixed_r, "imperfect, rectified by stereoRectifyUncalibrated")
rows.append(r)
points += p
np.save(os.path.join(TMP, "imperfect_H.npy"), np.stack([H1, H2]))

save_csv("vertical_misalignment.csv",
         ["pair", "matches", "mean |dy|, px", "median |dy|, px", "95th percentile |dy|, px",
          "picked points", "mean |dy| on picked, px", "median |dy| on picked, px",
          "max |dy| on picked, px"], rows)
save_csv("picked_points.csv", ["pair", "n", "xL", "yL", "xR", "yR", "|yL - yR|, px", "correlation"], points)

c = read_calib(os.path.join(DATA, "calib_imperfect.txt"))
print(f"dataset says for imperfect: dyavg {c.get('dyavg')}, dymax {c.get('dymax')} (full size)")

# crop of the imperfect pair before and after, lines every 25 px; the right crop is moved left by
# the median disparity of the matches so that both halves show the same details
y0, x0, ch, cw = 300, 450, 200, 300
dx = int(np.median(pa_i[:, 0] - pb_i[:, 0]))
before = lines(left_i[y0:y0 + ch, x0:x0 + cw], right_i[y0:y0 + ch, x0 - dx:x0 - dx + cw])
after = lines(fixed_l[y0:y0 + ch, x0:x0 + cw], fixed_r[y0:y0 + ch, x0 - dx:x0 - dx + cw])
cv2.imwrite(os.path.join(FIG, "imperfect_lines_crop.jpg"),
            cv2.resize(np.vstack([before, np.full((10, before.shape[1], 3), 255, np.uint8), after]), None,
                       fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
