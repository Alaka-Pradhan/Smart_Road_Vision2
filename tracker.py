from collections import defaultdict, deque
import time

class TrackHistory:
    """Stores recent center points per tracker ID for movement/event rules."""
    def __init__(self, maxlen=20):
        self.points = defaultdict(lambda: deque(maxlen=maxlen))
        self.last_seen = {}
        self.maxlen = maxlen

    def update(self, detections, frame_index):
        now = time.time()
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.points[tid].append((frame_index, center[0], center[1]))
            self.last_seen[tid] = now
        # Remove tracks not seen for 60 seconds.
        for tid in list(self.last_seen):
            if now - self.last_seen[tid] > 60:
                self.last_seen.pop(tid, None)
                self.points.pop(tid, None)

    def get(self, track_id):
        return list(self.points.get(track_id, []))
