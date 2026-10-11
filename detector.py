from ultralytics import YOLO
import numpy as np

class RoadObjectDetector:
    """YOLO detector/tracker adapter. Track IDs are returned when available."""
    def __init__(self, model_path="yolo11n.pt", confidence=0.35, iou=0.45,
                 image_size=640, tracker="bytetrack.yaml"):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tracker = tracker
        self.model = None

    def load(self):
        self.model = YOLO(self.model_path)
        return self

    def detect_and_track(self, frame):
        if self.model is None:
            self.load()
        results = self.model.track(
            frame, persist=True, conf=self.confidence, iou=self.iou,
            imgsz=self.image_size, tracker=self.tracker, verbose=False
        )
        result = results[0]
        detections = []
        if result.boxes is None:
            return detections, result.plot()
        names = result.names
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())
            detections.append({
                "bbox": coords,
                "class_id": cls_id,
                "label": str(names.get(cls_id, cls_id)),
                "confidence": conf,
                "track_id": track_id,
            })
        return detections, result.plot()


class AccidentDetector:
    """Specialized YOLOv11x detector fine-tuned for real-world traffic accident detection."""
    def __init__(self, model_path="models/accident_yolo11x.pt", confidence=0.25):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.model = None

    def load(self):
        import os
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        return self

    @property
    def is_available(self):
        import os
        return os.path.exists(self.model_path)

    def detect(self, frame):
        if not self.is_available:
            return []
        if self.model is None:
            self.load()
        if self.model is None:
            return []

        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        if not results or results[0].boxes is None:
            return detections
        names = results[0].names
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            # Enos-123 model classifies 'accident' and 'vehicle'
            if "accident" in label or cls_id == 0:
                detections.append({
                    "bbox": coords,
                    "class_id": cls_id,
                    "label": "Accident",
                    "confidence": conf,
                })
        return detections

