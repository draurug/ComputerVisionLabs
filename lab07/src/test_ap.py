# Lab 7, step 16 check: own NMS, matching, PR curve and AP on a small example worked out by hand

import numpy as np

from detection import average_precision, box_iou, match, nms, pr_curve

# two images, 3 ground truth boxes
gt = {"a": [(0, 0, 10, 20), (30, 0, 40, 20)], "b": [(0, 0, 10, 20)]}
# 6 detections (image, box, score), given out of order on purpose
dets = [
    ("b", (12, 0, 22, 20), 0.6),    # nothing there                          -> FP
    ("a", (0, 0, 10, 20), 0.9),     # exactly the first box of a             -> TP
    ("a", (1, 0, 11, 20), 0.8),     # same object again, its box is taken    -> FP
    ("a", (30, 2, 40, 22), 0.7),    # second box of a, IoU 0.82              -> TP
    ("b", (0, 10, 10, 30), 0.5),    # IoU 1/3 < 0.5                          -> FP
    ("b", (0, 1, 10, 21), 0.4),     # box of b, IoU 0.90                     -> TP
]

assert abs(box_iou((0, 10, 10, 30), (0, 0, 10, 20)) - 1 / 3) < 1e-12
tp, _ = match(dets, gt)
assert tp.tolist() == [False, True, False, True, False, True]

# by hand, sorted by score: TP FP TP FP FP TP
#   k:          1    2    3    4    5    6
#   precision:  1   1/2  2/3  1/2  2/5  1/2
#   recall:    1/3  1/3  2/3  2/3  2/3   1
# non-increasing precision: 1 at recall 1/3, 2/3 at recall 2/3, 1/2 at recall 1
# AP = 1/3 * 1 + 1/3 * 2/3 + 1/3 * 1/2 = 13/18 = 0.7222
recall, precision, _ = pr_curve([d[2] for d in dets], tp, n_gt=3)
assert np.allclose(precision, [1, 1 / 2, 2 / 3, 1 / 2, 2 / 5, 1 / 2])
assert np.allclose(recall, [1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1])
ap, _ = average_precision(recall, precision)
assert abs(ap - 13 / 18) < 1e-12, ap

# the same AP by a different route: integrate p_interp(r) = max{precision at recall >= r} on a fine grid
grid = np.linspace(0, 1, 300001)[1:]
dense = np.array([precision[recall >= r - 1e-12].max() for r in grid])
assert abs(dense.mean() - ap) < 1e-4

# without sorting by confidence (the typical mistake) the same detections give another number
r_bad, p_bad, _ = pr_curve([d[2] for d in dets], tp, n_gt=3, sort=False)
ap_bad, _ = average_precision(r_bad, p_bad)

# NMS is done per image. In image a the box at 0.8 overlaps the one at 0.9 with IoU 0.82 > 0.5 and is
# removed; in image b the two boxes on the object overlap with IoU 0.38 and both stay
in_a = [d for d in dets if d[0] == "a"]
in_b = [d for d in dets if d[0] == "b"]
keep_a = nms([d[1] for d in in_a], [d[2] for d in in_a], 0.5)
keep_b = nms([d[1] for d in in_b], [d[2] for d in in_b], 0.5)
assert [in_a[i][2] for i in keep_a] == [0.9, 0.7], keep_a
assert [in_b[i][2] for i in keep_b] == [0.6, 0.5, 0.4], keep_b
keep = [in_a[i][2] for i in keep_a] + [in_b[i][2] for i in keep_b]

print(f"own implementation: AP = {ap:.4f} (by hand 13/18 = {13 / 18:.4f}), "
      f"dense integration {dense.mean():.4f}, without sorting {ap_bad:.4f}; NMS keeps the scores {keep}")
print("all checks passed")
