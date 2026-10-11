from pathlib import Path
import cv2
import json
import re
from datetime import datetime
from config import EVIDENCE_DIR

class EvidenceManager:
    def __init__(self, output_dir=EVIDENCE_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_event(self, event, frame):
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", event["category"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.output_dir / f"{stamp}_{safe}.jpg"
        json_path = self.output_dir / f"{stamp}_{safe}.json"
        cv2.imwrite(str(image_path), frame)
        record = dict(event)
        record["evidence_image"] = str(image_path)
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return str(image_path), str(json_path)
