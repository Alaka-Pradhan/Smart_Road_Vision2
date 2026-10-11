"""
Test and demonstration script for Traffic Accident Detection using the fine-tuned YOLO11x model.
Evaluates the model on test surveillance accident images and generates annotated visualizations.
"""

import os
import cv2
from detector import AccidentDetector
from config import ACCIDENT_MODEL_PATH

def main():
    print("=" * 60)
    print("  TRAFFIC ACCIDENT DETECTION (YOLO11x) - TEST SUITE")
    print("=" * 60)

    detector = AccidentDetector(ACCIDENT_MODEL_PATH, confidence=0.25).load()
    if not detector.is_available or detector.model is None:
        print("[ERROR] Accident model weights not found at:", ACCIDENT_MODEL_PATH)
        return

    os.makedirs("test_accident_output", exist_ok=True)
    test_images = sorted([f for f in os.listdir("test_accident_data") if f.endswith(".jpg")])

    print(f"\nEvaluating {len(test_images)} test images from 'test_accident_data/'...\n")

    for fname in test_images:
        in_path = os.path.join("test_accident_data", fname)
        out_path = os.path.join("test_accident_output", f"annotated_{fname}")
        frame = cv2.imread(in_path)
        if frame is None:
            continue

        dets = detector.detect(frame)
        print(f"[{fname}] Detected {len(dets)} accident occurrence(s):")
        for i, d in enumerate(dets, 1):
            bbox = d["bbox"]
            conf = d["confidence"]
            print(f"   -> #{i}: Confidence {conf*100:.1f}%, Box: {bbox}")
            # Draw red neon accident box and banner
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            label = f"ACCIDENT {conf*100:.0f}%"
            cv2.putText(frame, label, (bbox[0], max(bbox[1]-12, 25)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(out_path, frame)
        print(f"   Saved annotated visualization to: {out_path}\n")

    print("All test evaluations completed successfully!")
    print("Visualizations available in: 'test_accident_output/'")

if __name__ == "__main__":
    main()
