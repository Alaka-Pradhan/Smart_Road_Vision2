import csv
from pathlib import Path
from config import EVENT_LOG, LOG_DIR

FIELDS = ["event_id", "timestamp", "category", "track_id", "object_class",
          "confidence", "frame_index", "details", "review_status",
          "evidence_image", "evidence_json"]

class EventLogger:
    def __init__(self, path=EVENT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, event):
        row = {field: event.get(field, "") for field in FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

    def update_review(self, event_id, status):
        if not self.path.exists():
            return
        with self.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        changed = False
        for row in rows:
            if row.get("event_id") == event_id:
                row["review_status"] = status
                changed = True
        if changed:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
