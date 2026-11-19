# Lab 6, part E: nearest-centroid tracker with and without Kalman, checked against the manual annotation

import os

import cv2
import numpy as np

from tracker import Tracker, blobs, masks
from utils import DATA, FIG, frame_with_label, hstack, read_clip, read_csv, save_csv, writer

np.random.seed(42)


def run(detections, kalman):
    # positions of all tracks on every frame: list of {id: (x, y, missed)}
    tracker = Tracker(kalman)
    out = []
    for centres in detections:
        out.append({t.id: (float(t.pos[0]), float(t.pos[1]), t.missed) for t in tracker.step(centres)})
    return out


def target_id(tracks, x, y, w, h):
    # Step 19. the tracker holds the object if a track centre is inside the annotated box;
    # the nearest such track is taken as the target's track
    best, best_d = None, np.inf
    for tid, (tx, ty, _) in tracks.items():
        if abs(tx - x) <= w / 2 and abs(ty - y) <= h / 2:
            d = np.hypot(tx - x, ty - y)
            if d < best_d:
                best, best_d = tid, d
    return best


def switches(ids):
    ids = [i for i in ids if i is not None]
    return sum(a != b for a, b in zip(ids, ids[1:]))


# Step 18. manual annotation: the woman in the red jacket, centre and size every 10th frame
ann = [r for r in read_csv(os.path.join(DATA, "annotations.csv")) if r["visible"] == "1"]
ann = [(int(r["clip frame"]), *(float(r[c]) for c in ("x", "y", "w", "h"))) for r in ann]
frames = read_clip("B")
detections = [blobs(mask)[0] for _, mask in masks(frames)]

# per-frame reference between the annotated frames: linear interpolation of the centre and the size
ref_frames = np.arange(ann[0][0], ann[-1][0] + 1)
ref = {f: [np.interp(f, [a[0] for a in ann], [a[i] for a in ann]) for i in range(1, 5)] for f in ref_frames}

rows = []
results = {}
for kalman in (False, True):
    label = "Kalman" if kalman else "simple"
    tracks = run(detections, kalman)
    ids = [target_id(tracks[f], x, y, w, h) for f, x, y, w, h in ann]
    ids_all = [target_id(tracks[f], *ref[f]) for f in ref_frames]
    held = np.mean([i is not None for i in ids])
    held_all = np.mean([i is not None for i in ids_all])
    first = ids[0]
    same = np.mean([i == first for i in ids])
    n_tracks = len({tid for t in tracks for tid in t})
    rows.append([label, len(ann), round(held, 3), round(same, 3), switches(ids),
                 round(held_all, 3), switches(ids_all), n_tracks])
    results[label] = (tracks, ids, ids_all)
    print(f"{label}: held on {held * 100:.0f} % of {len(ann)} annotated frames, with the first id "
          f"{same * 100:.0f} %, id switches {switches(ids)}; every frame: held {held_all * 100:.0f} %, "
          f"switches {switches(ids_all)}; tracks in the clip {n_tracks}")
    print("  ids on the annotated frames:", ids)
    changes = [(int(ref_frames[k + 1]), a, b) for k, (a, b) in enumerate(zip(ids_all, ids_all[1:])) if a != b]
    print("  changes frame by frame (frame, from, to):", changes)

save_csv("tracking.csv", ["tracker", "annotated frames", "held share", "held with the first id",
                          "id switches (annotated frames)", "held share (every frame, interpolated)",
                          "id switches (every frame)", "tracks in the clip"], rows)
save_csv("tracking_ids.csv", ["clip frame", "simple tracker id", "Kalman tracker id"],
         [[f] + [results[l][1][k] for l in ("simple", "Kalman")] for k, (f, *_) in enumerate(ann)])

# clip A: one person, the tracker should keep one id. The person = the largest blob of the frame,
# her track = the track closer than 20 px to it. In the first frames MOG2 is still learning the background
# (she is already in the first frame, so she is in the initial model and leaves a ghost)
detections_a, largest = [], []
for _, mask in masks(read_clip("A")):
    centres, boxes = blobs(mask)
    detections_a.append(centres)
    largest.append(centres[np.argmax(boxes[:, 2] * boxes[:, 3])] if len(centres) else None)
rows_a = []
for kalman in (False, True):
    label = "Kalman" if kalman else "simple"
    tracks = run(detections_a, kalman)
    ids = []
    for t, c in zip(tracks, largest):
        near = [(np.hypot(x - c[0], y - c[1]), tid) for tid, (x, y, _) in t.items()] if c is not None else []
        ids.append(min(near)[1] if near and min(near)[0] < 20 else None)
    main = max(set(ids) - {None}, key=ids.count)
    # from which frame the main id is there, and how often another id takes the person for a moment
    # (a piece of her blob starts a short track that is nearer to the blob centre for 1-2 frames)
    start = ids.index(main)
    after = ids[start:]
    breaks = sum(a == main and b != main for a, b in zip(after, after[1:]))
    share = after.count(main) / len(after)
    rows_a.append([label, len(ids), switches(ids[:start + 1]), start, round(share, 3), breaks])
    print(f"clip A, {label}: {switches(ids[:start + 1])} id changes before the main id {main} appears "
          f"at frame {start}, then it holds {share * 100:.0f} % of the frames, short breaks: {breaks}")
save_csv("tracking_clip_a.csv", ["tracker", "frames", "id changes before the main id", "main id from frame",
                                 "share with the main id after that", "short breaks"], rows_a)

# figures: annotation boxes and the track that holds the target, for both trackers
pics = []
for label in ("simple", "Kalman"):
    tracks, ids, _ = results[label]
    colors = {}
    pic = frames[ann[-1][0]].copy()
    for (f, x, y, w, h), tid in zip(ann, ids):
        cv2.rectangle(pic, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 255, 255), 1)
        if tid is not None:
            colors.setdefault(tid, tuple(int(c) for c in np.random.randint(0, 255, 3)))
    for tid, color in colors.items():
        path = [(int(t[tid][0]), int(t[tid][1])) for t in tracks if tid in t]
        cv2.polylines(pic, [np.array(path, np.int32)], False, color, 2)
        cv2.putText(pic, str(tid), path[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    pics.append(frame_with_label(pic, f"{label} tracker: boxes = annotation, lines = tracks on the target"))
cv2.imwrite(os.path.join(FIG, "tracking_B.jpg"), hstack(pics))

# annotation check: crops around every annotated box
crops = []
for f, x, y, w, h in ann:
    pic = frames[f].copy()
    cv2.rectangle(pic, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 1)
    cv2.circle(pic, (int(x), int(y)), 2, (0, 255, 0), -1)
    x0 = int(np.clip(x - 50, 0, pic.shape[1] - 100))
    y0 = int(np.clip(y - 70, 0, pic.shape[0] - 140))
    crop = cv2.resize(pic[y0:y0 + 140, x0:x0 + 100], None, fx=1.5, fy=1.5)
    cv2.putText(crop, str(f), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    crops.append(crop)
cv2.imwrite(os.path.join(FIG, "annotation_check.jpg"), hstack(crops[:8]) if len(crops) <= 8 else
            np.vstack([hstack(crops[:8]), hstack(crops[8:16] + [np.full_like(crops[0], 255)] * (16 - len(crops)))]))

# video with the Kalman tracks (coasting tracks in yellow)
tracks = results["Kalman"][0]
out = writer("tracking_kalman.avi", frames[0].shape)
for k, frame in enumerate(frames):
    pic = frame.copy()
    for tid, (x, y, missed) in tracks[k].items():
        color = (0, 255, 255) if missed else (255, 255, 0)
        cv2.circle(pic, (int(x), int(y)), 4, color, -1)
        cv2.putText(pic, str(tid), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out.write(pic)
out.release()
