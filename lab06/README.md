# Lab 6. Optical flow and object tracking

Lucas-Kanade on corner points with the forward-backward check, dense Farneback flow, MOG2 background
subtraction with counting people that cross a line, own nearest-centroid tracker with and without a Kalman
filter, checked against a manual annotation.

## Data

There were no own videos from a fixed camera, so both clips are cut from the OpenCV sample `vtest.avi`
(pedestrians in front of a building, static camera, 768×576, 10 fps, 795 frames). The video is not in the
repo, `prepare_data.py` downloads it into `data/`.

| Clip | Frames of vtest.avi | Duration | Area | What is in it |
|---|---|---|---|---|
| A | 605–709 | 10.5 s | x 0–440, y 270–576 (440×306) | one woman walking on the lawn |
| B | 420–619 | 20.0 s | whole frame | several people at a time; the woman in red passes a man and then goes behind the sign on the lamp post |

In clip A the legs of a passer-by on the road touch the top right corner in frames 683–702.

`data/annotations.csv` (step 18): centre and size of the woman in red in every 10th frame of clip B
(20 frames, she is visible in 15 of them: clip frames 10–150). The annotation was done by visual inspection
of enlarged frames with a coordinate grid, with AI assistance; `figures/annotation_check.jpg` shows the boxes.

## Run

```bash
source ../.venv/bin/activate
cd src
python prepare_data.py   # download vtest.avi, clip sizes, step 2 brightness check
python sparse_flow.py    # part B: LK, forward-backward check, survival plot, trajectories video
python dense_flow.py     # part C: Farneback, HSV pictures, time per frame compared with LK
python background.py     # part D: MOG2, cleaning, blobs, counting people crossing a line
python tracking.py       # part E: tracker with and without Kalman against the annotation
```

Each script takes 1–4 minutes. Videos are written to `results/tmp/` (not in the repo).

## Notes

- Step 2: the dataset camera has fixed exposure. Mean brightness over clip B is 118.7 ± 0.81 (whole frame)
  and 120.4 ± 0.62 for the background pixels only, clip A 86.4 ± 0.44; the max − min over a clip is below
  3.5 levels of 255 (`results/clips.csv`, `figures/brightness.png`).
- Sparse flow: `goodFeaturesToTrack(300, 0.01, 10)`, `calcOpticalFlowPyrLK` with a 21×21 window and
  `maxLevel = 2` (three pyramid levels: the image and two halvings), forward-backward limit 1 px, a lost
  point stays lost. Survival to the end (`results/sparse_survival.csv`, `figures/sparse_survival.png`):
  clip B 300 → 260 points (86.7 %) without the filter and 300 → 170 (56.7 %) with it; clip A (the lawn has
  little texture, only 79 corners) 70 (88.6 %) and 36 (45.6 %). Without the filter many points slide along
  the grass and the road: 131 of 300 points in clip B end more than 5 px from their start, with the filter
  only 36 move that far before they are dropped and only 4 of them survive to the end. So the filtered set
  is mostly the static background; a point on a walking person is lost at the first occlusion, turn of
  the body or arm swing. Four levels instead of three give almost the same: 170 survivors in B, 34 instead
  of 36 in A (people move less than ~10 px per frame here, three levels are enough).
- Dense flow: Farneback with pyr_scale 0.5, 3 levels, window 15, 3 iterations, poly_n 5, poly_sigma 1.2.
  HSV picture: hue = direction, value = |v| / 8 px (the same scale for all frames, so frames without motion
  stay dark; hue 0° red = to the right, 180° cyan = to the left, image y goes down). About 5.5 % of the
  pixels move faster than 1 px per frame in both clips. Time per frame, both methods on the same pairs,
  one OpenCV thread (`results/timing.csv`): clip B (768×576) Farneback 629 ms, LK forward + backward on 300
  points 42 ms; clip A (440×306) 278 ms and 16 ms, so dense flow is 15–18 times slower. The absolute
  numbers are inflated: the machine was under heavy load during the run (load average 25–40 on 8 cores),
  only the ratio should be trusted. Sparse flow is enough when only the motion of a few objects is needed
  (tracking, counting, stabilisation); dense flow is needed when every pixel matters: motion segmentation,
  the shape of a moving region. In clip A only a few corners sit on the woman (hair, legs), while the
  dense flow outlines her whole silhouette (`figures/farneback_A.jpg`); inside her uniform black coat both
  methods see almost no motion (aperture problem).
- Without the check, points on the grass and the road next to a passing person are dragged along with
  her and leave long false trails, with the check they are dropped (`figures/trajectories_A.jpg`,
  `figures/trajectories_B.jpg`, left without, right with the check).
- Background subtraction: MOG2(history 500, varThreshold 16, detectShadows), shadow pixels (127) set to 0,
  opening 3×3, closing 5×5, blobs from `connectedComponentsWithStats` with area ≥ 150 px. After frame 20
  the median blob is 55 px and 57 % of the blobs are below 150 px (`figures/blob_areas.png`, two clear
  groups). Whole people are 700–9000 px (700–1300 px far on the road); noise from the curb and the
  fluttering tape is mostly 10–100 px. The threshold is kept low on purpose: pieces of a person cut by an
  occluder or by the morphology are 140–550 px (her legs under the sign at frame 58 are 547 px) and
  should stay. A few tape pieces above 150 px remain, they are far from the counting line.
- Counting (steps 14, 15): horizontal line y = 300 px across clip B, a track is counted once, when its
  matched centroid goes from one side of the line to the other. Manual count by looking through every
  3rd frame: 4 crossings (the woman up, two men from the top down, one man from the bottom right up).
  Automatic: simple tracker 6 (4 up, 2 down), Kalman tracker 5 (4 up, 1 down)
  (`results/line_count.csv`, `results/line_events.csv`). 3 crossings are found correctly by both. The
  errors: two men walking towards each other merge into one blob, one of them gets a new track only when
  they separate, already below the line, so his crossing is lost; a man in dark clothes standing right at
  the line is split into a torso and a legs blob, the torso blob "crosses" up; two people entering from the
  right border at the height of the line give partial blobs whose centroid jumps across it (frames 82 and
  150; the simple tracker counts both, the Kalman one only the second).
- Tracker: prediction = position + velocity (the simple one smooths the velocity as 0.5·old + 0.5·last
  step; the Kalman one has state (x, y, vx, vy), constant velocity, measurement noise 3 px), greedy
  matching by the smallest distance within 40 px, an unmatched track moves along the prediction and is
  deleted after 5 frames without a match.
- Tracking against the annotation (`results/tracking.csv`, `results/tracking_ids.csv`,
  `figures/tracking_B.jpg`): a track holds the target if its centre is inside the annotated box. Both
  trackers hold the woman on all 15 annotated frames. The simple tracker keeps her first id only on 5 of 15
  (33 %), 1 switch on the annotated frames and 3 changes frame by frame: when a man passes right behind
  her (clip frames 41–43) the ids swap for two frames and come back; at frames 52–57 another man walks
  in front of her exactly while her upper body is behind the sign, and the two make one blob. Only one
  track can take that blob, the other one moves along its prediction. In the simple tracker the man's track
  takes it, hers has no match for 6 frames (52–57), one more than allowed, is deleted, and at frame 58 (her legs appear
  under the sign as a separate 550 px blob) she gets a new id 13. With Kalman the merged blob goes to her
  track for the first 3 frames and to his for the next 3, each coasts at most 3 frames, and both keep their
  ids: the same id on 100 % of the frames, 0 switches. So the Kalman tracker wins here partly because of
  which track got the blob first, not only because its prediction is better; one episode is not enough to
  separate the two.
- Clip A, one person (`results/tracking_clip_a.csv`): in the first frames MOG2 is still learning the
  background (she is already in the first frame, so she is part of the initial model and leaves a ghost),
  the ids change while it settles: the main id appears at frame 5 with the simple tracker and at frame 16
  with Kalman. After that it holds 94 % and 97 % of the frames, with 3 and 2 short breaks of 1–2 frames
  where a piece of her blob starts its own short track.

## Output

- `results/` — clips and brightness, sparse survival, timing, blobs, line count and events, tracking
  metrics and ids
- `figures/` — brightness, survival curves, trajectories (A, B), Farneback (A, B), MOG2 steps, blob areas,
  annotation check, tracks on the target for both trackers
- `results/tmp/` (not in the repo) — trajectories, Farneback, counting and tracking videos
