# Lab 5. Stereo pair: rectification, disparity, depth

Variant 3 of the task: there is no stereo camera, so the pair is the Middlebury 2014 scene "Piano" with
ground truth disparity. Q from `stereoRectify`, vertical misalignment, StereoBM and StereoSGBM on a grid
of parameters, depth and point cloud, zones of systematic errors.

Data (`data/`): `im0.png`, `im1.png` — the "perfect" pair, `im0_imperfect.png`, `im1_imperfect.png` — the
same scene with a slightly wrong rectification, all downscaled to 1410×960 (half size, `INTER_AREA`);
`disp0_gt.png`, `disp1_gt.png` — ground truth for the left and right image, 16-bit, value = disparity × 64,
0 = unknown; `calib.txt`, `calib_imperfect.txt` — the original full-size calibration.

## Run

```bash
source ../.venv/bin/activate
cd src
python prepare_data.py <folder>   # only to rebuild data/: downloads Piano-perfect/imperfect.zip (~165 MB)
python rectify.py      # part B: Q, lines every 25 px, vertical misalignment, imperfect pair fixed
python disparity.py    # part C: BM and SGBM grid, 72 combinations (about 15 minutes)
python best_map.py     # step 9: best maps of both methods
python depth.py        # part D: depth, check against the ground truth, point cloud
python errors.py       # part E: error zones
```

`rectify.py` and `best_map.py` save `Q` and the best maps to `results/tmp/` (not in the repo), `depth.py`
and `errors.py` read them.

## Notes

- Calibration at half size: f = 1413.09 px, B = 178.09 mm, doffs = cx1 − cx0 = 61.885 px, ground truth
  disparity 19.3…104.3 px (ndisp = 130), depth 1.51…3.10 m. Depth: Z = f·B / (d + doffs).
- Part B: the pair is already rectified, so there is no `stereoCalibrate`; `stereoRectify` with the
  dataset K0, K1, R = I, T = (−B, 0, 0) gives R1 = R2 = I, the maps move no pixel (max shift 0.000 px),
  and Q gives back exactly f, B and doffs (`results/rectify_q.csv`). `reprojectImageTo3D` with this Q
  and the formula agree to 1e-6 m.
- Step 6, |yL − yR|: instead of marking by hand, 15 corners (the strongest Shi–Tomasi corner per cell of
  a 5 × 3 grid, `cornerSubPix`) matched by correlation of a 21 × 21 patch with a parabola peak, kept only
  if correlation ≥ 0.9 and the search back returns within 1 px; also over all SIFT matches (ratio 0.7,
  mutual, RANSAC on F). Perfect pair: 0.131 px mean over 1431 matches, 15 corners median 0.09 px.
  Imperfect pair: 0.374 px (the dataset gives dyavg 0.408 px at full size). After
  `stereoRectifyUncalibrated` from the matches: 0.123 px. All below 1 px; the single corners give
  up to 2–3 px on depth edges and the carpet, so the median of the 15 is used, not the max.
- Part C: both methods on gray images, speckleWindowSize 100, speckleRange 2, disp12MaxDiff 1,
  SGBM P1 = 8·bs², P2 = 32·bs². Coverage = share of valid pixels, bad-1 = share of |d − d_gt| > 1 px among
  pixels valid in both, "correct" = share of the ground truth pixels found within 1 px (used to choose the
  best map, because bad-1 alone prefers an almost empty map). Best BM: nd 128, bs 21, ur 5 — coverage
  64.3 %, bad-1 12.5 %, correct 57.2 %, 0.39 s. Best SGBM: nd 128, bs 5, ur 5 — coverage 82.4 %, bad-1
  21.1 %, correct 66.0 %, about 6 s. Times are medians of 3 runs, but the machine was busy with another
  job, so SGBM times jump between 1 and 10 s for similar settings; BM is about 10 times faster.
- numDisparities 64 is too small: the near objects have d up to 104 px, SGBM bad-1 is 37–42 % (21–30 % with 128). 192
  is no better than 128 and the strip x < nd on the left, where the matcher has no candidates, gets wider.
  Larger blockSize gives BM more coverage (31 → 56 %) and more errors (3.8 → 13.3 %); uniquenessRatio
  5 → 15 lowers both coverage and bad-1 by a few percent.
- Surprise: SGBM on the color images with the usual P1 = 8·3·bs², P2 = 32·3·bs² breaks from blockSize 13
  (bad-1 50 %, 78 %, 100 % for 13, 15, 21; `results/sgbm_color_vs_gray.csv`): the costs are 16-bit and
  overflow. On gray images with P1 = 8·bs², P2 = 32·bs² it works up to 21.
- Step 11 (no tape measure in variant 3): 7 objects, 15 × 15 window median, ground truth depth vs SGBM
  depth (`results/depth_objects.csv`): 5 textured objects 0.02–0.38 % (0.4–7.9 mm), the two walls are gross
  mismatches, 8.8 % and 4.7 %. Over all pixels in 8 depth bins (`results/depth_error_bins.csv`): the
  absolute error goes from 3.3 mm at 1.7 m to 12.5 mm at 2.8 m, the relative one from 0.19 % to 0.46 %;
  it follows Z²/(f·B)·|Δd|, but the mean |Δd| differs between bins, so the curve is not a clean line.
  One pixel of disparity is 11.5 mm at 1.7 m and 33.4 mm at 2.9 m.
- Point cloud: 1 353 600 pixels → 1 114 861 valid → 1 094 589 in 1.60…2.97 m → 1 077 858 after removing
  isolated points (mean distance to 8 neighbours > mean + 2 std). Saved to `results/tmp/cloud.ply`.
- Part E (`results/error_zones.csv`, `figures/error_zones.png`), bad-1 for BM / SGBM: textureless
  (std < 2 in 15 × 15, mostly the walls, 19 % of the image) — BM leaves 67 % empty and is right where it
  answers (9.5 %), SGBM fills 74 % but 50.5 % of it is wrong; occlusions at left borders of objects — 84 /
  86 %; shiny red vase and brass lamp — 36 / 37 %; the rest of the image — 9.4 / 11.1 %. The piano keys
  were expected to fail as a repeated pattern but did not (9.4 / 15.8 %): the groups of 2 and 3 black keys
  break the period and the keyboard is slanted, so there is one best match.

## Output

- `results/` — Q check, vertical misalignment and the 15 points, disparity grid, parameter effects, best
  maps, color vs gray SGBM, depth of the objects, error vs depth, point cloud sizes, error zones
- `figures/` — lines on the rectified pair, imperfect pair before/after, grid plot, best maps with ground
  truth and error, depth map, depth error vs distance, point cloud, error zones
