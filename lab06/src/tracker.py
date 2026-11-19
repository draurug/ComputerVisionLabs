# background subtraction -> blobs, and a nearest-centroid tracker with or without a Kalman filter

import cv2
import numpy as np

# Step 11-13
MOG2 = dict(history=500, varThreshold=16, detectShadows=True)
MIN_AREA = 150      # px, whole people are 700+ px, noise is mostly below 100, pieces of a split person 140-550
OPEN = np.ones((3, 3), np.uint8)
CLOSE = np.ones((5, 5), np.uint8)

# Step 16
GATE = 40           # px, a detection farther than this from the prediction is not matched
MAX_MISSED = 5      # frames without a match before the track is deleted


def masks(frames):
    # mask of every frame: raw MOG2 output (255 object, 127 shadow, 0 background) and the cleaned one
    mog = cv2.createBackgroundSubtractorMOG2(**MOG2)
    for frame in frames:
        raw = mog.apply(frame)
        # Step 12. shadows (127) go to the background
        mask = np.where(raw == 255, 255, 0).astype(np.uint8)
        # Step 13. opening removes single pixels, closing fills small holes inside people
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE)
        yield raw, mask


def blobs(mask, min_area=MIN_AREA):
    # connected components bigger than min_area: centroids (N, 2) and boxes (N, 4: x, y, w, h)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return centroids[keep].astype(np.float32), stats[keep, :4]


class Track:
    def __init__(self, track_id, pos, kalman):
        self.id = track_id
        self.pos = np.array(pos, np.float32)
        self.vel = np.zeros(2, np.float32)
        self.missed = 0
        self.kf = None
        if kalman:
            # Step 17. state (x, y, vx, vy), constant velocity, measurement (x, y)
            kf = cv2.KalmanFilter(4, 2)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
            kf.processNoiseCov = np.diag([1, 1, 0.5, 0.5]).astype(np.float32)
            kf.measurementNoiseCov = np.diag([9, 9]).astype(np.float32)      # blob centre jitters by ~3 px
            kf.errorCovPost = np.diag([10, 10, 25, 25]).astype(np.float32)
            kf.statePost = np.array([[pos[0]], [pos[1]], [0], [0]], np.float32)
            self.kf = kf

    def predict(self):
        if self.kf is not None:
            return self.kf.predict()[:2].ravel()
        return self.pos + self.vel

    def update(self, pred, det):
        if det is None:
            # no detection: move along the prediction
            # (KalmanFilter.predict already copies the predicted state to statePost)
            self.pos = pred
            self.missed += 1
            return
        if self.kf is not None:
            state = self.kf.correct(det.reshape(2, 1).astype(np.float32)).ravel()
            self.pos, self.vel = state[:2], state[2:]
        else:
            # velocity = smoothed last step
            self.vel = 0.5 * self.vel + 0.5 * (det - self.pos)
            self.pos = det.copy()
        self.missed = 0


class Tracker:
    def __init__(self, kalman=False):
        self.kalman = kalman
        self.tracks = []
        self.next_id = 1

    def step(self, detections):
        # Step 16. predict every track, match greedily by the smallest distance, update, add, delete
        preds = [t.predict() for t in self.tracks]
        pairs = []
        if self.tracks and len(detections):
            d = np.linalg.norm(np.array(preds)[:, None, :] - detections[None, :, :], axis=2)
            while True:
                i, j = np.unravel_index(np.argmin(d), d.shape)
                if d[i, j] > GATE:
                    break
                pairs.append((i, j))
                d[i, :] = np.inf
                d[:, j] = np.inf
        matched_t = {i: j for i, j in pairs}
        matched_d = set(matched_t.values())
        for i, (track, pred) in enumerate(zip(self.tracks, preds)):
            track.update(pred, detections[matched_t[i]] if i in matched_t else None)
        for j in range(len(detections)):
            if j not in matched_d:
                self.tracks.append(Track(self.next_id, detections[j], self.kalman))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return self.tracks
