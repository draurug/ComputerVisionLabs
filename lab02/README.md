# Lab 2. Camera calibration, undistortion

Chessboard 9×6 inner corners (10×7 squares) printed on A4, square size measured with a ruler: 15.5 mm.
25 photos with a phone, 1920×2560, in `data/calib/` (`names.csv` maps them to the original file names).

## Run

```bash
source ../.venv/bin/activate
cd src
python calibrate.py     # steps 6-11, 15: corners, calibration, per-frame errors, camera_params.yml
python experiments.py   # step 12: error and fx vs number of frames
python undistort.py     # steps 13-14: alpha = 0 / 1, straight lines before and after
```

`calibrate.py` has to run first, the other two read `results/corners.npz` and `results/camera_params.yml`.

## Notes

- `findChessboardCorners` on the full-size photo finds the board on 3 frames only; on a 2× or 4× downscaled
  copy it finds 22, the rest are found with `findChessboardCornersSB`. The corners are then refined
  with `cornerSubPix` on the full-size image (`utils.find_corners`).
- Error per frame is computed with `projectPoints`, outliers = frames with error above 2 × median
  (1 frame dropped).
- Final mean reprojection error is about 1.1 px, above the 0.5 px reference. The main reason is the sheet:
  it is not glued to anything rigid and the rows of corners are visibly bent in the same way before and
  after undistortion (`figures/straight_lines_crop.jpg`). The photos also went through Telegram, so they
  are recompressed and have no EXIF.
- Distortion coefficients are small (k1 ≈ 0.05): the phone corrects the lens in software, so the frames
  with alpha = 0 and alpha = 1 look almost the same.
- There was no separate frame with a doorway or a table edge for step 14, so the straight-line check
  uses the rows of the board: a line through the first and last corner of a row, max deviation of the
  corners in between.

## Output

- `results/camera_params.yml` — camera matrix, distortion, image size, error, number of frames, date
- `results/*.csv` — detection, subpix effect, per-frame errors, calibration numbers, N experiment, lines
- `figures/` — corners example, per-frame error bars, error/fx vs N, alpha = 0/1, lines before/after
