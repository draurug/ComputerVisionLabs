# Lab 7. Segmentation, classical detector, quality metrics

Otsu, marker watershed, SLIC superpixels and GrabCut on 10 own images, IoU and Dice against own masks;
why pixel accuracy is misleading for a small object; agreement between two annotations; the HOG + SVM
people detector on 30 Penn-Fudan images with own NMS, matching, precision–recall curve and AP.

## Data

`data/images/` — 10 own photos (iPhone 14 Pro, downscaled to 1024 px on the long side), one object each:

| Image | Source | Object | Share of the image |
|---|---|---|---|
| honey_window, honey_lamp | lab 1, `IMG_3013`, `IMG_3014` | honey bottle (window light / lamp) | 22.8 %, 22.5 % |
| hoop | lab 1, `IMG_2308` | embroidery hoop with the frame | 74.6 % |
| sheet_01, sheet_05, sheet_09 | lab 2, `calib_01/05/09` | sheet with the chessboard (with the board under it) | 36.0 %, 64.1 %, 36.2 % |
| house_0663, house_0664, house_2217, house_0252 | lab 1 | the house (small, far away) | 1.9–2.6 % |

`data/annotations/*.json` — polygons of the objects (labelme-like: a list of polygons in pixels),
traced by visual inspection of enlarged crops with a coordinate grid, with AI assistance.
`make_masks.py` fills them into `data/masks/*.png` (object 255, background 0) and checks step 2: the same size
as the image and exactly two values. Conventions: the cap of the bottle is part of the object, the fabric
hanging out of the hoop and the hand are not; the houses are traced down to the ground line, the low bushes
in front of the walls are included, the chimney is included, the fences and annexes are not.

`data/annotations/second/` — a second, independent tracing of two images (step 9). There was no colleague
to exchange with, so the second annotation is a separate pass by the same author at a different zoom, made
without looking at the first polygon. It underestimates the real disagreement between two people.

Part E: [Penn-Fudan Pedestrian Database](https://www.cis.upenn.edu/~jshi/ped_html/) (170 images, boxes of the
pedestrians), `prepare_data.py` downloads it into `data/PennFudanPed/` (not in the repo). 30 images evenly
spread over the sorted list are used, 78 annotated pedestrians.

## Run

```bash
source ../.venv/bin/activate
cd src
python make_masks.py     # part A: polygons -> masks, step 2 check
python segmentation.py   # parts B, C: 4 methods, IoU/Dice table, accuracy trap (about 1 minute)
python agreement.py      # part D: IoU between the two annotations
python prepare_data.py   # part E data: download Penn-Fudan (54 MB)
python test_ap.py        # step 16 check of the own NMS / matching / PR / AP on a worked example
python hog_detector.py   # part E: HOG + SVM, NMS, PR curve, AP (about 10 seconds)
```

## Notes

- One rule for all four methods, nothing is tuned on the masks: a method gives two classes and the class that
  covers less of the image border is taken as the object.
- Otsu: `cv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)`. Watershed: the 7 steps of the task (Otsu,
  opening 3×3 ×2, `distanceTransform` L2 5, sure object = distance > 0.5·max, sure background = inverted
  dilation 3×3 ×3, markers from `connectedComponents`, `watershed`). SLIC: `n_segments=400, compactness=10`,
  the mean Lab colour of the superpixels is split into two classes by k-means (k = 2, 10 attempts, seed 42).
  GrabCut: rectangle = bounding box of the own mask plus 5 % on each side (stands in for the rectangle the
  user draws), 5 iterations, object = GC_FGD | GC_PR_FGD.
- Results (`results/segmentation.csv`, `results/segmentation_summary.csv`, `figures/iou_per_image.png`),
  IoU mean ± std [min, max] over 10 images: GrabCut 0.841 ± 0.195 [0.370, 0.998], SLIC 0.467 ± 0.376
  [0.022, 0.814], Otsu 0.402 ± 0.329 [0.018, 0.704], watershed 0.295 ± 0.287 [0.000, 0.638]. The spread
  comes from two groups: on the 6 large objects SLIC gets 0.757 ± 0.040 and Otsu 0.653 ± 0.073, on the 4
  houses all methods without a rectangle are below 0.05 — they split sky from ground, not the house.
  GrabCut on the houses is 0.661 ± 0.201 (worst: house_2217, 0.37: only the dark roof is kept, the white walls look like the pale sky and the white
  fence outside the rectangle and go to the background).
- Otsu on the sheets is 0.69–0.70: the black squares of the chessboard go to the background class.
  Watershed on the hoop is 0.146: the fabric texture is full of strong gradients, the flooding from the
  background marker wins over most of the unknown ring.
- Step 8 (`results/accuracy_trap.csv`): on the houses (1.9–2.6 % of the image) the empty mask has pixel
  accuracy 97.4–98.1 % and IoU 0. Otsu on house_0663 has accuracy 70 % and IoU 0.03.
- Steps 9, 10 (`results/agreement.csv`, `figures/agreement.jpg`): IoU between the two annotations is 0.969 for
  the honey bottle (sharp edges) and 0.921 for the house (bushes, eaves). GrabCut on the bottle (0.956) is
  already at this level.
- HOG: `cv2.HOGDescriptor()` (64×128, cells 8×8, blocks 16×16 with stride 8, 9 bins, 3780 values),
  default people SVM, `detectMultiScale(hitThreshold=-0.5, winStride=(8, 8), padding=(8, 8), scale=1.05,
  groupThreshold=0)` — raw windows with scores, 2924 in 30 images. The low `hitThreshold` only adds weak
  windows to the end of the ranking. Own NMS (greedy, IoU > 0.5 removed), own matching (from the most
  confident detection, best IoU in the same image, TP if ≥ 0.5 and the box is free), PR curve and AP with the
  non-increasing precision (all-point area), `src/detection.py`.
- AP check (`test_ap.py`): lecture 8 is not among the course materials I have (only lectures 1–2), so the
  example with AP = 0.65 could not be reproduced. Instead a worked example (6 detections, 3 objects,
  duplicate, IoU 1/3 miss) gives 13/18 = 0.7222 by hand, the code gives the same, and a dense numerical
  integration of the envelope agrees to 1e-4.
- The window has a margin around the person: annotated boxes have width/height 0.37 (median), the window 0.5.
  With the boxes as returned only 59 % of the pedestrians have a detection with IoU ≥ 0.5 and AP = 0.122
  (NMS 0.5). Shrinking the boxes the way the OpenCV sample `peopledetect.py` draws them (15 % of the width
  and 5 % of the height off each side; not tuned here) raises that to 85 % and AP to 0.530 — this is the
  main result (`results/detection_summary.csv`, `results/pr_curve.csv`, `figures/pr_curve.png`).
  Best F1 0.534 at score ≥ 0.65 (precision 0.574, recall 0.500).
- NMS threshold (shrunk boxes): none 0.250 (2924 detections), 0.3 → 0.532 (306), 0.5 → 0.530 (430), 0.7 →
  0.475 (750); OpenCV `groupRectangles` (groupThreshold 2) 0.468. Without NMS every extra window on the same
  person is a FP; with a loose threshold (0.7) neighbouring windows survive and become FPs.
- Without sorting by score the same detections give AP 0.162 in the order they were produced and
  0.169 ± 0.018 [0.132, 0.227] over 200 random orders.
- Most high-score FPs are half-body windows (legs or torso of a person who already has a TP) and windows
  between two people walking side by side.

## Output

- `results/` — masks, per-image segmentation metrics and summary, accuracy trap, annotator agreement,
  detection summary, PR curve, all detections with their match, extra AP numbers
- `figures/` — annotation, segmentation examples, IoU per image, SLIC superpixels, agreement, HOG, NMS effect,
  PR curves, detection examples
