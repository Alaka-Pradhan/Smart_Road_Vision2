from pathlib import Path

APP_NAME = "Smart Road Vision"
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
EVIDENCE_DIR = BASE_DIR / "evidence"
MODEL_DIR = BASE_DIR / "models"
EVENT_LOG = LOG_DIR / "events.csv"

# General-purpose YOLO weights can detect common road vehicles, but not
# every specialist class (helmet status, signal state, emergency vehicles).
DEFAULT_MODEL = "yolov8n.pt" if (BASE_DIR / "yolov8n.pt").exists() else "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
PROCESS_EVERY_N_FRAMES = 1
DENSITY_LOW_MAX = 5
DENSITY_MEDIUM_MAX = 15
WRONG_WAY_MIN_MOVEMENT_PX = 45
WRONG_WAY_EXPECTED_DIRECTION = "down"  # down/up/left/right in image coordinates
EVENT_COOLDOWN_SECONDS = 8
MAX_TRACK_HISTORY = 20

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Traffic Accident Detection Module (fine-tuned YOLO11x)
ACCIDENT_MODEL_PATH = MODEL_DIR / "accident_yolo11x.pt"
ACCIDENT_CONF_THRESHOLD = 0.25
ACCIDENT_COOLDOWN_SECONDS = 10
ACCIDENT_CHECK_INTERVAL = 3  # Check every N frames to ensure smooth video FPS
