# Smart Road Vision
### AI-assisted traffic monitoring, candidate-event alerting, evidence capture, and review dashboard

This project is a runnable prototype based on the supplied research-paper draft. It provides a modular OpenCV + YOLO tracking pipeline and a Tkinter desktop dashboard.

## Implemented in this prototype
- Open a webcam (camera index `0`) or prerecorded video.
- YOLO object detection with persistent ByteTrack IDs.
- Vehicle classes: car, motorcycle, bus, truck (dependent on selected model labels).
- Per-frame vehicle count and configurable heuristic density category.
- A transparent wrong-way **candidate** rule based on track movement and a configurable expected direction.
- Timestamped candidate-event logs in `logs/events.csv`.
- A JPG evidence snapshot and JSON event record in `evidence/` for each candidate event.
- Tkinter dashboard with video overlay, status, count, density, FPS, recent events, and manual review statuses.

## Important scope limits
This version does **not** claim that every module in the paper is fully implemented:
- Red-light crossing needs a configured stop line and a reliable signal-state source; this prototype does not automatically detect the traffic-light state or declare red-light violations.
- Helmet detection needs a specialist helmet model/dataset; generic YOLO weights are not sufficient.
- Accident alerts: Integrated fine-tuned YOLO11x model (`Enos-123/traffic-accident-detection-yolo11x`) trained on the Roboflow Accident dataset, providing real-time accident bounding-box localization, live alerts, and automatic evidence capture.
- Emergency-vehicle recognition only works if the selected model contains suitable trained classes; generic COCO weights typically do not provide a dedicated ambulance/fire-engine class.
- Density thresholds are heuristics, not a validated measure of physical road density.
- It does not control traffic lights, issue legal fines, or perform automatic number-plate OCR.

Treat alerts as decision support. A human must review evidence. Do not connect this prototype to public traffic-signal infrastructure.

## Requirements
- Python 3.10 or 3.11 recommended.
- Windows, Linux, or macOS with a working GUI environment.
- Internet access the first time the default YOLO weights are downloaded, unless you supply a local model file.
- A webcam is optional; a prerecorded road video is enough.

## Setup (Windows PowerShell)
```powershell
cd Smart_Road_Vision
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main_gui.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_gui.py
```

## Setup (Linux/macOS)
```bash
cd Smart_Road_Vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main_gui.py
```

## Run it
1. Enter `0` to use the default webcam, or click **Browse video** and select an MP4/AVI/MOV/MKV file.
2. Leave model as `yolo11n.pt` for a lightweight default model, or enter a local weights path.
3. Press **Start**. On first run, Ultralytics may download weights.
4. Press **Stop** to end processing.
5. Review candidate events in the table. Select a row and set its review status.
6. Use **Open logs** and **Evidence** to inspect generated files.

## Project structure
```text
Smart_Road_Vision/
├── main_gui.py           # Tkinter dashboard and processing thread
├── video_processor.py    # OpenCV source capture
├── detector.py           # YOLO inference + ByteTrack
├── tracker.py            # Track-history store
├── event_rules.py        # Transparent event rules/count/density
├── evidence_manager.py   # Evidence JPG + JSON
├── logger.py             # CSV event log and review updates
├── config.py             # Thresholds and paths
├── requirements.txt
├── README.md
├── logs/
└── evidence/
```

## Configuration
Edit `config.py` to change:
- confidence and IoU thresholds,
- density thresholds,
- expected direction (`down`, `up`, `left`, `right`),
- minimum movement for a wrong-way candidate,
- event cooldown.

The current direction rule is global and must be calibrated for the camera view. It can produce false alerts on turning traffic, camera-angle changes, track-ID switches, or noisy detections.

## Evaluation checklist aligned with the research paper
Create a source-video-separated test set and manually label:
- vehicles and classes,
- track/count ground truth,
- positive and negative wrong-way clips,
- difficult lighting, weather, occlusion, and distant objects.

Report:
- detector precision, recall, F1, mAP@0.5 and mAP@0.5:0.95,
- count error and tracking ID switches (if measured),
- event precision/recall/F1 and false alerts per minute,
- inference latency, end-to-end latency, effective FPS, dropped frames,
- CPU/GPU/memory use and evidence/logging reliability.

Do not copy metrics from cited papers as if they were results from this project. Fill the research paper's result tables only after running tests.

## Troubleshooting
- **`ModuleNotFoundError: ultralytics`**: activate the environment and run `pip install -r requirements.txt`.
- **Model download fails**: download a compatible Ultralytics `.pt` weight file separately and enter its local path in the Model field.
- **Camera won't open**: close other apps using it, check Windows camera permissions, or use a video file.
- **GUI opens but no frames appear**: confirm the selected file is a supported video and readable by OpenCV.
- **Slow performance**: use `yolo11n.pt`, reduce `IMAGE_SIZE` in `config.py`, or process every second frame (requires adding frame skipping logic).
- **Wrong-way alerts look incorrect**: set the correct `WRONG_WAY_EXPECTED_DIRECTION` for your camera and review event snapshots.
