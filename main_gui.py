import os
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from config import (APP_NAME, DEFAULT_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE, LOG_DIR, EVIDENCE_DIR,
                    ACCIDENT_MODEL_PATH, ACCIDENT_CONF_THRESHOLD, ACCIDENT_CHECK_INTERVAL)
from video_processor import VideoProcessor
from detector import RoadObjectDetector, AccidentDetector
from tracker import TrackHistory
from event_rules import EventAnalyzer
from evidence_manager import EvidenceManager
from logger import EventLogger

class SmartRoadVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME + " | AI Traffic Monitoring")
        self.root.geometry("1320x820")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#101820")
        self.running = False
        self.worker = None
        self.frame_queue = queue.Queue(maxsize=2)
        default_src = "sample_traffic.mp4" if os.path.exists("sample_traffic.mp4") else "0"
        self.source_var = tk.StringVar(value=default_src)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.conf_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        self.status_var = tk.StringVar(value="Idle")
        self.signal_var = tk.StringVar(value="Unknown")
        self.count_var = tk.StringVar(value="0")
        self.density_var = tk.StringVar(value="—")
        self.fps_var = tk.StringVar(value="0")
        self.last_alert_var = tk.StringVar(value="None")
        self.accident_enabled_var = tk.BooleanVar(value=True)
        self.accident_status_var = tk.StringVar(value="Normal / Clear")
        self.events = []
        self.current_photo = None
        self.logger = EventLogger()
        self._build_ui()
        self.root.after(80, self._poll_frames)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=7)
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        header = tk.Frame(self.root, bg="#172b3a", height=72)
        header.pack(fill="x")
        tk.Label(header, text="SMART ROAD VISION", bg="#172b3a", fg="#f5f7fa",
                 font=("Segoe UI", 21, "bold")).pack(side="left", padx=22, pady=15)
        tk.Label(header, text="AI-assisted traffic monitoring • Human review required",
                 bg="#172b3a", fg="#9fb3c8", font=("Segoe UI", 10)).pack(side="left", padx=8)

        controls = tk.Frame(self.root, bg="#101820")
        controls.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(controls, text="Video file or camera index:", bg="#101820", fg="white").pack(side="left")
        tk.Entry(controls, textvariable=self.source_var, width=35).pack(side="left", padx=6)
        ttk.Button(controls, text="Browse video", command=self.browse_video).pack(side="left", padx=3)
        ttk.Button(controls, text="Demo Video", command=self.load_sample_video).pack(side="left", padx=3)
        tk.Label(controls, text="YOLO model:", bg="#101820", fg="white").pack(side="left", padx=(15, 0))
        tk.Entry(controls, textvariable=self.model_var, width=17).pack(side="left", padx=5)
        ttk.Button(controls, text="Start", command=self.start).pack(side="left", padx=3)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(controls, text="Open logs", command=lambda: self._open_path(LOG_DIR)).pack(side="left", padx=3)
        ttk.Button(controls, text="Evidence", command=lambda: self._open_path(EVIDENCE_DIR)).pack(side="left", padx=3)
        ttk.Checkbutton(controls, text="Accident AI (YOLO11x)", variable=self.accident_enabled_var).pack(side="left", padx=6)

        body = tk.Frame(self.root, bg="#101820")
        body.pack(fill="both", expand=True, padx=14, pady=8)
        left = tk.Frame(body, bg="#172b3a", width=285)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="LIVE DASHBOARD", bg="#172b3a", fg="#57d9a3",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(15, 10))
        self._metric(left, "System status", self.status_var)
        self._metric(left, "Vehicles in current frame", self.count_var)
        self._metric(left, "Estimated traffic density", self.density_var)
        self._metric(left, "Processing FPS", self.fps_var)
        self._metric(left, "Last candidate alert", self.last_alert_var)
        self._metric(left, "Accident Safety Status", self.accident_status_var)
        tk.Label(left, text="Signal state (manual input)", bg="#172b3a", fg="#d9e2ec").pack(anchor="w", padx=14, pady=(15, 4))
        ttk.Combobox(left, textvariable=self.signal_var, values=["Unknown", "Red", "Yellow", "Green"],
                     state="readonly", width=17).pack(anchor="w", padx=14)
        tk.Label(left, text="Note: signal state is not detected automatically.", bg="#172b3a",
                 fg="#9fb3c8", wraplength=245, justify="left").pack(anchor="w", padx=14, pady=8)
        tk.Label(left, text="Confidence threshold", bg="#172b3a", fg="#d9e2ec").pack(anchor="w", padx=14, pady=(10, 2))
        ttk.Scale(left, from_=0.1, to=0.9, variable=self.conf_var, orient="horizontal").pack(fill="x", padx=14)

        right = tk.Frame(body, bg="#172b3a")
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="LIVE VIDEO / DETECTIONS", bg="#172b3a", fg="#f5f7fa",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=9)
        self.video_label = tk.Label(right, bg="#0b1117", fg="#9fb3c8",
                                    text="Choose a video file or camera index 0, then press Start.",
                                    font=("Segoe UI", 12))
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        bottom = tk.Frame(self.root, bg="#101820")
        bottom.pack(fill="both", padx=14, pady=(0, 14))
        tk.Label(bottom, text="RECENT EVENTS — verify each alert before acting",
                 bg="#101820", fg="#f5f7fa", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        cols = ("timestamp", "category", "track", "confidence", "review", "evidence")
        self.event_tree = ttk.Treeview(bottom, columns=cols, show="headings", height=7)
        for col, label, width in [
            ("timestamp","Timestamp",155), ("category","Event",190), ("track","Track ID",80),
            ("confidence","Confidence",95), ("review","Review status",110), ("evidence","Evidence image",360)]:
            self.event_tree.heading(col, text=label)
            self.event_tree.column(col, width=width, anchor="w")
        self.event_tree.pack(fill="both", expand=True)
        review_row = tk.Frame(bottom, bg="#101820")
        review_row.pack(fill="x", pady=5)
        ttk.Button(review_row, text="Mark selected: Confirmed", command=lambda: self.review_selected("Confirmed")).pack(side="left", padx=(0, 5))
        ttk.Button(review_row, text="Mark selected: Rejected", command=lambda: self.review_selected("Rejected")).pack(side="left", padx=5)
        ttk.Button(review_row, text="Mark selected: Needs review", command=lambda: self.review_selected("Needs review")).pack(side="left", padx=5)

    def _metric(self, parent, label, variable):
        card = tk.Frame(parent, bg="#20394a")
        card.pack(fill="x", padx=10, pady=5)
        tk.Label(card, text=label, bg="#20394a", fg="#9fb3c8",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(7, 1))
        tk.Label(card, textvariable=variable, bg="#20394a", fg="#f5f7fa",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10, pady=(0, 8))

    def browse_video(self):
        path = filedialog.askopenfilename(title="Select traffic video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.source_var.set(path)

    def load_sample_video(self):
        for candidate in ["sample_traffic.mp4", "real_traffic_highway.mp4"]:
            if os.path.exists(candidate):
                self.source_var.set(candidate)
                return
        messagebox.showinfo("Demo Video", "No sample video file found in current folder.")

    def start(self):
        if self.running:
            return
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("Missing source", "Choose a video file or camera index.")
            return
        self.running = True
        self.status_var.set("Starting…")
        self.worker = threading.Thread(target=self._process_video, args=(source,), daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.status_var.set("Stopping…")

    def _process_video(self, source):
        processor = None
        try:
            processor = VideoProcessor(source).open()
            detector = RoadObjectDetector(
                self.model_var.get().strip() or DEFAULT_MODEL,
                confidence=float(self.conf_var.get()), iou=IOU_THRESHOLD, image_size=IMAGE_SIZE
            ).load()
            history = TrackHistory()
            analyzer = EventAnalyzer()
            evidence = EvidenceManager()
            accident_detector = AccidentDetector(ACCIDENT_MODEL_PATH, ACCIDENT_CONF_THRESHOLD)
            if self.accident_enabled_var.get() and accident_detector.is_available:
                try:
                    accident_detector.load()
                except Exception:
                    pass

            start_time = time.time()
            processed = 0
            self._put({"kind": "status", "value": "Running"})
            while self.running:
                ok, frame = processor.read()
                if not ok:
                    break
                processed += 1
                detections, annotated = detector.detect_and_track(frame)
                history.update(detections, processor.frame_index)

                accident_dets = []
                if (self.accident_enabled_var.get() and
                    accident_detector.is_available and
                    processed % ACCIDENT_CHECK_INTERVAL == 0):
                    try:
                        accident_dets = accident_detector.detect(frame)
                    except Exception:
                        pass

                analysis = analyzer.analyze(
                    detections, history, annotated, processor.frame_index,
                    signal_state=self.signal_var.get(),
                    accident_detections=accident_dets
                )

                if accident_dets:
                    for acc in accident_dets:
                        ax1, ay1, ax2, ay2 = acc["bbox"]
                        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 255), 4)
                        cv2.putText(annotated, f"ACCIDENT DETECTED {acc['confidence']*100:.0f}%",
                                    (ax1, max(ay1 - 10, 25)), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

                saved_events = []
                for event in analysis["events"]:
                    image_path, json_path = evidence.save_event(event, annotated)
                    event["evidence_image"] = image_path
                    event["evidence_json"] = json_path
                    self.logger.log(event)
                    saved_events.append(event)
                elapsed = max(time.time() - start_time, 0.001)
                fps = processed / elapsed
                self._put({
                    "kind": "frame", "frame": annotated,
                    "count": analysis["vehicle_count_current"],
                    "density": analysis["density"], "fps": fps,
                    "events": saved_events,
                    "accident_detected": bool(accident_dets)
                })
            self._put({"kind": "status", "value": "Stopped" if self.running is False else "Finished"})
        except Exception as exc:
            self._put({"kind": "error", "value": str(exc)})
        finally:
            if processor:
                processor.release()
            self.running = False

    def _put(self, item):
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(item)
        except queue.Empty:
            pass
        except queue.Full:
            pass

    def _poll_frames(self):
        try:
            while True:
                item = self.frame_queue.get_nowait()
                if item["kind"] == "status":
                    self.status_var.set(item["value"])
                elif item["kind"] == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Smart Road Vision", item["value"])
                elif item["kind"] == "frame":
                    frame = item["frame"]
                    self.count_var.set(str(item["count"]))
                    self.density_var.set(item["density"])
                    self.fps_var.set(f'{item["fps"]:.1f}')
                    if item.get("accident_detected"):
                        self.accident_status_var.set("CRITICAL: ACCIDENT DETECTED")
                    elif not any(e.get("category") == "Traffic Accident" for e in self.events[:3]):
                        self.accident_status_var.set("Normal / Clear")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    max_w, max_h = 920, 440
                    image.thumbnail((max_w, max_h))
                    self.current_photo = ImageTk.PhotoImage(image)
                    self.video_label.configure(image=self.current_photo, text="")
                    for event in item["events"]:
                        self.last_alert_var.set(event["category"])
                        self._add_event(event)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_frames)

    def _add_event(self, event):
        self.events.append(event)
        self.event_tree.insert("", 0, iid=event["event_id"], values=(
            event.get("timestamp", ""), event.get("category", ""),
            event.get("track_id", ""), f'{event.get("confidence", 0):.2f}',
            event.get("review_status", "Pending"), event.get("evidence_image", "")
        ))

    def review_selected(self, status):
        selected = self.event_tree.selection()
        if not selected:
            messagebox.showinfo("Select event", "Select an event row first.")
            return
        for event_id in selected:
            self.logger.update_review(event_id, status)
            values = list(self.event_tree.item(event_id, "values"))
            values[4] = status
            self.event_tree.item(event_id, values=values)
            for event in self.events:
                if event["event_id"] == event_id:
                    event["review_status"] = status

    @staticmethod
    def _open_path(path):
        os.makedirs(path, exist_ok=True)
        try:
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showinfo("Folder location", f"Open this folder manually:\n{path}\n\n{exc}")

    def on_close(self):
        self.running = False
        self.root.after(200, self.root.destroy)

def main():
    root = tk.Tk()
    SmartRoadVisionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
