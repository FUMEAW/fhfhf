"""Vision pipeline for the AutoSort system.

The detector exposes clean structured results instead of raw arrays, and it
defensively handles camera failures and low-confidence detections.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from autosort.config import LABEL_ALIASES

YOLO = None


def _load_yolo_class():
    """Import ultralytics only when the model is actually needed."""
    global YOLO
    if YOLO is not None:
        return YOLO
    try:
        from ultralytics import YOLO as _YOLO
    except Exception:  # pragma: no cover - import-time guard for missing/unsupported deps.
        return None
    YOLO = _YOLO
    return YOLO


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[int, int]
    class_id: int
    slot_num: Optional[int] = None


class YoloDetector:
    def __init__(self, camera_index: Optional[int] = 0, model_path: Optional[str] = None, conf_threshold: float = 0.45):
        base_dir = Path(__file__).resolve().parent.parent
        # Try common model file names in order of preference
        for candidate in ("weights.onnx", "best.onnx", "model.onnx", "model.pt", "best.pt"):
            candidate_path = base_dir / "vision_stuff" / "ai_model" / candidate
            if candidate_path.exists():
                default_model = candidate_path
                break
        else:
            default_model = base_dir / "vision_stuff" / "ai_model" / "weights.onnx"  # will fail clearly if missing
        self.model_path = Path(model_path) if model_path else default_model
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.camera_available = False

        yolo_cls = _load_yolo_class()
        self.model = None
        self.cap = None
        self._frame = None
        self._frame_lock = threading.Lock()

        self._detections: List[DetectionResult] = []
        self._detections_lock = threading.Lock()

        self._running = False
        self._capture_thread = None
        self._inference_thread = None

    @staticmethod
    def find_camera_index(max_index: int = 5) -> Optional[int]:
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                return idx
        return None

    def start(self) -> None:
        """Open the camera and start the background capture/inference threads."""
        yolo_cls = _load_yolo_class()
        if yolo_cls is None:
            raise RuntimeError("ultralytics is not installed; the vision pipeline cannot run.")

        if self.model is None:
            self.model = yolo_cls(str(self.model_path))

        if self.camera_index is None:
            self.camera_index = self.find_camera_index()

        if self.camera_index is None:
            self.camera_available = False
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.camera_available = False
            return

        self.camera_available = True
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self) -> None:
        """Stop capture and release camera resources."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_available = False

    def get_detections(self) -> List[DetectionResult]:
        with self._detections_lock:
            return list(self._detections)

    def get_frame(self):
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @staticmethod
    def get_slot(center_x: int, center_y: int, width: int = 1, height: int = 1) -> Optional[int]:
        if width <= 0 or height <= 0:
            return None

        mid_x = width / 2.0
        mid_y = height / 2.0
        if center_x < mid_x and center_y < mid_y:
            return 3
        if center_x >= mid_x and center_y < mid_y:
            return 2
        if center_x < mid_x and center_y >= mid_y:
            return 4
        if center_x >= mid_x and center_y >= mid_y:
            return 1
        return None

    def _capture_loop(self) -> None:
        while self._running and self.cap is not None:
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.02)
                continue
            with self._frame_lock:
                self._frame = frame

    def _inference_loop(self) -> None:
        while self._running:
            with self._frame_lock:
                frame = None if self._frame is None else self._frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            if self.model is None:
                self.model = _load_yolo_class()(str(self.model_path))

            try:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            except Exception:
                time.sleep(0.05)
                continue

            detections: List[DetectionResult] = []
            height, width, _ = frame.shape
            for box in results.boxes:
                if len(box.xyxy) == 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                class_id = int(box.cls[0].item())
                raw_label = self.model.names[class_id]
                normalized = (raw_label or "").lower().strip()
                mapped = LABEL_ALIASES.get(normalized)
                if mapped is None:
                    continue
                confidence = float(box.conf[0].item())
                if confidence < self.conf_threshold:
                    continue

                detections.append(
                    DetectionResult(
                        label=mapped,
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        center=(center_x, center_y),
                        slot_num=self.get_slot(center_x, center_y, width=width, height=height),
                    )
                )

            with self._detections_lock:
                self._detections = detections


if __name__ == "__main__":
    detector = YoloDetector(camera_index=0, conf_threshold=0.45)
    detector.start()
    try:
        while True:
            frame = detector.get_frame()
            if frame is None:
                continue

            for det in detector.get_detections():
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, det.center, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{det.label} {det.confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("YOLO preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.stop()
        cv2.destroyAllWindows()
