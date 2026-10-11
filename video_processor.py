import cv2
import time

class VideoProcessor:
    """OpenCV wrapper for file paths, webcam indices, and frame acquisition."""
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.frame_index = 0
        self.started_at = None

    def open(self):
        source = self.source
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Could not open video source: {self.source}")
        self.frame_index = 0
        self.started_at = time.time()
        return self

    def read(self):
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if ok:
            self.frame_index += 1
        return ok, frame

    @property
    def fps(self):
        if self.capture is None:
            return 0.0
        return self.capture.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def resolution(self):
        if self.capture is None:
            return (0, 0)
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
