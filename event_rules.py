import math
import time
from config import (VEHICLE_CLASSES, DENSITY_LOW_MAX, DENSITY_MEDIUM_MAX,
                    WRONG_WAY_MIN_MOVEMENT_PX, WRONG_WAY_EXPECTED_DIRECTION,
                    EVENT_COOLDOWN_SECONDS)

class EventAnalyzer:
    """
    Transparent candidate-event rules. These are screening heuristics, not
    legal determinations or a validated accident classifier.
    """
    def __init__(self):
        self.last_event_time = {}
        self.counted_tracks = set()
        self.total_vehicle_count = 0
        self.previous_positions = {}

    def _cooldown_ok(self, key):
        now = time.time()
        if now - self.last_event_time.get(key, 0) < EVENT_COOLDOWN_SECONDS:
            return False
        self.last_event_time[key] = now
        return True

    @staticmethod
    def _center(det):
        x1, y1, x2, y2 = det["bbox"]
        return ((x1+x2)//2, (y1+y2)//2)

    def analyze(self, detections, history, frame, frame_index, signal_state="Unknown", accident_detections=None):
        events = []
        vehicles = [d for d in detections if d["label"] in VEHICLE_CLASSES]
        unique_ids = {d["track_id"] for d in vehicles if d.get("track_id") is not None}
        # Counting uses a track once, avoiding a count on every frame.
        for det in vehicles:
            tid = det.get("track_id")
            if tid is not None and tid not in self.counted_tracks:
                self.counted_tracks.add(tid)
                self.total_vehicle_count += 1

        density_count = len(unique_ids) if unique_ids else len(vehicles)
        if density_count <= DENSITY_LOW_MAX:
            density = "Low"
        elif density_count <= DENSITY_MEDIUM_MAX:
            density = "Medium"
        else:
            density = "High"

        # Check for wrong-way traffic
        for det in vehicles:
            tid = det.get("track_id")
            if tid is None:
                continue
            points = history.get(tid)
            if len(points) < 4:
                continue
            _, x0, y0 = points[0]
            _, x1, y1 = points[-1]
            dx, dy = x1-x0, y1-y0
            expected = WRONG_WAY_EXPECTED_DIRECTION
            opposite = ((expected == "down" and dy < -WRONG_WAY_MIN_MOVEMENT_PX) or
                        (expected == "up" and dy > WRONG_WAY_MIN_MOVEMENT_PX) or
                        (expected == "left" and dx > WRONG_WAY_MIN_MOVEMENT_PX) or
                        (expected == "right" and dx < -WRONG_WAY_MIN_MOVEMENT_PX))
            key = ("wrong_way", tid)
            if opposite and self._cooldown_ok(key):
                events.append(self._event("Wrong-way candidate", det, frame_index,
                                          f"Track {tid} moved opposite configured direction ({expected})."))

        # Check for traffic accidents
        if accident_detections:
            for acc in accident_detections:
                key = "traffic_accident_alert"
                if self._cooldown_ok(key):
                    conf_pct = acc.get("confidence", 0.0) * 100
                    events.append(self._event(
                        "Traffic Accident",
                        acc,
                        frame_index,
                        f"Traffic accident / collision detected with {conf_pct:.1f}% confidence."
                    ))

        return {
            "events": events,
            "vehicle_count_current": density_count,
            "vehicle_count_total": self.total_vehicle_count,
            "density": density,
        }

    @staticmethod
    def _event(category, det, frame_index, details):
        return {
            "event_id": f"{int(time.time()*1000)}-{det.get('track_id','na')}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "category": category,
            "track_id": det.get("track_id"),
            "object_class": det.get("label"),
            "confidence": det.get("confidence", 0.0),
            "frame_index": frame_index,
            "details": details,
            "review_status": "Pending",
        }
