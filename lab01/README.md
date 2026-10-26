# Lab 1. Image representation, color spaces

How an image is stored in memory (dtype, channel order, compression), and in which
color space the thresholds for an object are the most stable when the lighting changes.

## Run

```bash
source ../.venv/bin/activate
cd src
python part_a.py    # steps 1-5: array properties, BGR/RGB, uint8 overflow, JPEG, EXIF
python part_b.py    # steps 8-11: thresholds in RGB / HSV / Lab
python part_c.py    # steps 12-14: aliasing, interpolation, gamma, gray world
```

The masks for part B and the dataset frames are already in `data/`. To make them again:

```bash
python make_mask.py object_light1.jpg object_mask.png 360 380 800 1330
python make_mask.py object_light2.jpg object_mask2.png 310 430 820 1330
python extract_frames.py <path to IMG_1332.MOV> 10 rot
```

## Data

| File | What it is |
|---|---|
| `photo.jpg` | landscape, iPhone 14 Pro (parts A and C) |
| `texture.jpg` | cross-stitch embroidery, fine regular grid (step 12) |
| `object_light1.jpg` | jar of honey on a blue chair, daylight |
| `object_light2.jpg` | the same under a lamp (the phone moved a bit) |
| `object_mask.png`, `object_mask2.png` | reference masks (GrabCut) |
| `dataset/rot_01..10.jpg` | 10 frames of a camera rotation, from the video IMG_1332.MOV (no EXIF) |
| `dataset/scene_IMG_*.jpg` | 10 photos of the same scene, different angles and lighting (with EXIF) |

The video is not in the repo. `dataset/rot_frames.csv` has the frame numbers and video metadata.

## Output

- `results/` — CSV tables
- `figures/` — plots
