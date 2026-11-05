# Lab 4. Feature matching, RANSAC, panorama

SIFT matching with the ratio test, why plain least squares fails on raw matches, own RANSAC for the
homography, threshold study, panorama from 5 frames and two cases where it breaks.

Data: the 10 frames of the camera rotation video from lab 1 (`rot_01..10.jpg`), downscaled to 540×960.

## Run

```bash
source ../.venv/bin/activate
cd src
python matching.py   # parts A, B: ratio test, mutual check, raw matches picture, least-squares H
python ransac.py     # part C: own RANSAC, iterations formula, comparison with cv2, tau study
python panorama.py   # parts D, E: panorama from rot_01..05, parallax and small overlap cases
```

## Notes

- Pair for parts A–C: `rot_03` and `rot_04`. T = 0.8 gives 744 candidates, the mutual check leaves 681.
  Least squares on all 744 (`findHomography` with method 0) gives a mean error of 122 px and a badly
  warped frame (`figures/lsq_warp.jpg`).
- Own RANSAC (τ = 3, 1000 iterations): 396 inliers, 1.27 px on them; `cv2.findHomography` with RANSAC:
  409 inliers, 1.23 px. With the observed outlier share of 0.47 the formula gives 55 iterations for
  p = 0.99, so 1000 is more than enough.
- τ study (`figures/ransac_threshold.png`): the number of inliers grows all the way to τ = 8 and the
  error on them grows with it; after τ = 8 the count is flat at about 700 of 744, which means almost all
  candidates, outliers included, are counted. τ = 3 is used for the panorama.
- Panorama (`figures/panorama_rotation.jpg`): frames `rot_01..05`, `rot_03` is the reference, the
  homographies of the neighbouring pairs are chained to it. Blending: each frame has a weight that goes
  to zero at its border, the canvas is the weighted mean, which gives a linear transition in the
  overlap. Mean error on the inliers over the 4 pairs: 1.08 px.
- Failure 1, parallax (`figures/panorama_parallax.jpg`): at the end of the video a pole is close to the
  camera. The pair errors are still about 1 px because RANSAC fits the background, but the pole is
  doubled in the panorama: the near and the far objects need different homographies.
- Failure 2, small overlap (`results/small_overlap.csv`): `rot_03` with `rot_04` gives 744 candidates and
  409 inliers, with `rot_05` only 52 and 13, with `rot_07` 39 and 7. With a few inliers the error looks
  small but the homography is not reliable.

## Output

- `results/` — ratio test table, least squares error, RANSAC vs cv2, tau study, panorama pair errors,
  small overlap table
- `figures/` — raw matches, least-squares warp, inliers/outliers, tau plots, two panoramas
