# Lab 3. Detectors and descriptors of local features

Own Harris detector, comparison with `cv2.cornerHarris`, SIFT and ORB, repeatability of the three
detectors under rotation, scale, brightness and noise.

Images: 4 frames from the lab 1 dataset, downscaled to 1024 px on the long side (`data/`).

## Run

```bash
source ../.venv/bin/activate
cd src
python harris.py          # parts A, B: own Harris vs cornerHarris, half-size test
python features.py        # part C: SIFT and ORB, keypoints and descriptors
python repeatability.py   # part D: repeatability grid, 4 plots (about 2 minutes)
```

## Notes

- Harris: Sobel 3×3, products, GaussianBlur σ = 1.5, R = det − 0.04·tr², threshold 0.01·max,
  3×3 local maxima, then greedy non-maximum suppression with r = 10 px, up to 1000 points (`utils.py`).
- Comparison with `cornerHarris` (step 8): with the gaussian window from the task the agreement is
  about 70 %, because `cornerHarris` sums the products over a square `blockSize` window, not a gaussian,
  and the suppression with r = 10 is sensitive to small changes of the ranking. With the same 5×5 box
  window the agreement is 100 % on all images, so the implementation itself is correct
  (`results/harris_vs_library.csv`, both columns).
- Half-size test (step 10): only 12–15 % of the original Harris points are found again.
- Same conditions for all detectors: 1000 points max, the same suppression radius r = 10 (applied to SIFT
  and ORB keypoints by their response too), ε = 3 px, 4 images, the result is the mean over the images.
- Repeatability: a point of P₁ moved by H counts if it is inside the new image and a P₂ point is closer
  than ε; each P₂ point is counted once; divided by min(|P₁|, |P₂|).
- Results (`results/repeatability.csv`, `figures/repeatability_*.png`): SIFT at 30° is 71 % (reference
  > 60 %). At scale 0.5 Harris (65 %) is higher than SIFT (22 %), not lower as the reference says. The
  metric compares positions only: a corner stays a corner at any scale, so its position repeats, while
  SIFT points here sit mostly on the fine texture of trees, where the blobs merge after downscaling and
  the top-1000 set changes. The scale sensitivity of Harris shows up in step 10, where the points are
  compared in the original coordinates and divided by the full-size count.

## Output

- `results/` — Harris vs library, half-size test, SIFT/ORB table, repeatability grid
- `figures/` — Harris points and response map, rich keypoints for SIFT and ORB, 4 repeatability plots
