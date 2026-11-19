# Computer Vision Systems (part 1) — labs

## Setup

Python 3.15, Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` — main libraries with pinned versions,
`requirements-full.txt` — full `pip freeze`.
scikit-image has no stable wheel for Python 3.15 yet, so the 0.27.0 release candidate is pinned
(it is used only in lab 1).

OpenCV 5.0 (`opencv-contrib-python`).

## Layout

```
labNN/
├── README.md     what the lab does and how to run it
├── src/          code
├── data/         input images
├── results/      numbers (CSV)
└── figures/      plots and pictures
```

Videos are not stored in the repo (see `.gitignore`).

## Labs

- `lab01/` — image representation, color spaces
- `lab02/` — camera calibration, undistortion
- `lab03/` — local features: Harris, SIFT, ORB, repeatability
- `lab04/` — matching, RANSAC, panorama
- `lab05/` — stereo: rectification, disparity, depth (Middlebury 2014)
- `lab06/` — optical flow, background subtraction, tracking
