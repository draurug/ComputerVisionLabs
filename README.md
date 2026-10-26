# Computer Vision Systems (part 1) — labs

## Setup

Python 3.14, Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` — main libraries with pinned versions,
`requirements-full.txt` — full `pip freeze`.

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
