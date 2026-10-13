"""Vision pipeline for the AutoSort system.

Uses onnxruntime InferenceSession directly — no ultralytics dependency.
Session startup is explicit (start_session) and inference is a clean
run_inference call that returns structured DetectionResult objects.

Preprocessing follows the reference implementation:
  - Resize to model input size (default 640x640)
  - BGR -> RGB
  - Normalize to [0, 1] float32
  - Transpose to CHW and add batch dimension
  - Run session, parse YOLOv8/v11 output tensor
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from onnxruntime import InferenceSession

from autosort.config import LABEL_ALIASES


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in original pixels
    center: Tuple[int, int]
    class_id: int
    slot_num: Optional[int] = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YoloDetector:
    """Threaded ONNX-based detector for AutoSort.

    Usage:
        detector = YoloDetector(camera_index=2, conf_threshold=0.45)
        detector.start()          # opens camera + inference threads
        detections = detector.get_detections()
        frame = detector.get_frame()
        detector.stop()
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        input_size: int = 640,
    ):
        base_dir = Path(__file__).resolve().parent.parent

        # Search for an existing model file in order of preference
        default_model: Optional[Path] = None
        for candidate in ("weights.onnx", "best.onnx", "model.onnx"):
            p = base_dir / "vision_stuff" / "ai_model" / candidate
            if p.exists():
                default_model = p
                break

        if default_model is None:
            # Fall through to a clear error at session-start time
            default_model = base_dir / "vision_stuff" / "ai_model" / "weights.onnx"

        self.model_path = Path(model_path) if model_path else default_model

        # Load class names from class_names.txt, skipping index 0 (background placeholder)
        class_names_path = base_dir / "vision_stuff" / "ai_model" / "class_names.txt"
        self._class_names: List[str] = []
        if class_names_path.exists():
            lines = class_names_path.read_text().strip().splitlines()
            self._class_names = [line.strip() for line in lines]

        self.camera_index = camera_index
        self.conf_threshold = float(conf_threshold)
        self.input_size = int(input_size)
        self.camera_available = False

        # Session state — populated by start_session()
        self._sess: Optional[InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._num_classes: int = 0
        self._orig_w: int = self.input_size
        self._orig_h: int = self.input_size

        # Camera + threading state
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._detections: List[DetectionResult] = []
        self._detections_lock = threading.Lock()

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Session management — mirrors start_detect_session() from reference
    # ------------------------------------------------------------------

    def start_session(self) -> bool:
        """Load the ONNX model and start an InferenceSession.

        Returns True on success.  Safe to call multiple times (no-op if already loaded).
        """
        if self._sess is not None:
            return True

        model_path = str(self.model_path)
        if not self.model_path.exists():
            print(f"ONNX model not found: {model_path}")
            return False

        try:
            self._sess = InferenceSession(model_path)
            self._input_name = self._sess.get_inputs()[0].name
            self._output_names = [o.name for o in self._sess.get_outputs()]
            # Infer number of classes from output shape: (1, 4+num_classes, 8400)
            out_shape = self._sess.get_outputs()[0].shape
            if len(out_shape) == 3:
                self._num_classes = int(out_shape[1]) - 4
            print(
                f"ONNX session started: {self.model_path.name} | "
                f"input={self._input_name} | "
                f"outputs={self._output_names} | "
                f"classes={self._num_classes}"
            )
            return True
        except Exception as exc:
            print(f"Error loading ONNX model: {exc}")
            self._sess = None
            return False

    # ------------------------------------------------------------------
    # Preprocessing — mirrors onnx_classify.py normalize+transpose pattern
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → float32 CHW batch tensor normalized to [0, 1]."""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)  # (1, 3, H, W)

    # ------------------------------------------------------------------
    # Inference — mirrors sess.run() pattern from reference
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run one inference pass on a BGR frame. Returns DetectionResult list."""
        if self._sess is None:
            return []

        self._orig_h, self._orig_w = frame.shape[:2]
        img_data = self._preprocess(frame)

        try:
            results = self._sess.run(
                self._output_names,
                {self._input_name: img_data},
            )
        except Exception as exc:
            print(f"Inference error: {exc}")
            return []

        # YOLOv8/v11 ONNX output: (1, 4 + num_classes, 8400) — transpose to (8400, 4+nc)
        raw = np.array(results[0])
        if raw.ndim == 3:
            raw = raw[0].T  # (8400, 4+nc)

        return self._parse_output(raw)

    # ------------------------------------------------------------------
    # Output parsing — anchor-free YOLOv8/v11 style
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _parse_output(self, out: np.ndarray) -> List[DetectionResult]:
        """Parse the transposed (8400, 4+nc) output tensor."""
        boxes_cx = out[:, 0]
        boxes_cy = out[:, 1]
        boxes_w  = out[:, 2]
        boxes_h  = out[:, 3]
        class_logits = out[:, 4:]

        # Convert cx/cy/w/h from model-pixel coords to original image coords
        scale_x = self._orig_w / self.input_size
        scale_y = self._orig_h / self.input_size

        x1 = (boxes_cx - boxes_w / 2) * scale_x
        y1 = (boxes_cy - boxes_h / 2) * scale_y
        x2 = (boxes_cx + boxes_w / 2) * scale_x
        y2 = (boxes_cy + boxes_h / 2) * scale_y

        # Confidence = max class score (sigmoid for YOLOv8/v11 — no obj score)
        scores = self._softmax(class_logits)
        best_class = np.argmax(scores, axis=1)
        best_score = scores[np.arange(len(scores)), best_class]

        # Filter below confidence floor
        keep = best_score >= self.conf_threshold
        if not keep.any():
            return []

        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        best_class = best_class[keep]
        best_score = best_score[keep]

        # NMS
        boxes_nms = np.stack([x1, y1, x2, y2], axis=1).tolist()
        scores_nms = best_score.tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.conf_threshold, iou_threshold=0.45)
        if len(indices) == 0:
            return []

        indices = [int(i) for i in np.array(indices).flatten()]
        detections: List[DetectionResult] = []

        for idx in indices:
            class_id = int(best_class[idx])
            confidence = float(best_score[idx])
            bx1, by1, bx2, by2 = float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])
            center_x = int((bx1 + bx2) / 2)
            center_y = int((by1 + by2) / 2)

            # Map class index to label using class_names.txt
            # Index 0 is a background placeholder — skip it
            if class_id == 0:
                continue
            if self._class_names and class_id < len(self._class_names):
                raw_label = self._class_names[class_id]
            else:
                raw_label = str(class_id)
            normalized = raw_label.lower().strip()
            mapped = LABEL_ALIASES.get(normalized, normalized)

            detections.append(DetectionResult(
                label=mapped,
                class_id=class_id,
                confidence=confidence,
                bbox=(bx1, by1, bx2, by2),
                center=(center_x, center_y),
                slot_num=self.get_slot(center_x, center_y, self._orig_w, self._orig_h),
            ))

        return detections

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

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
        return 1

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the ONNX session, open camera, start background threads."""
        if not self.start_session():
            print("WARNING: ONNX session failed to start. Vision pipeline offline.")

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
        """Stop threads and release camera."""
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

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while self._running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
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

            detections = self._run_inference(frame)

            with self._detections_lock:
                self._detections = detections

            time.sleep(0.05)  # ~20 fps inference cap


# ---------------------------------------------------------------------------
# Standalone preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = YoloDetector(camera_index=2, conf_threshold=0.45)
    detector.start()
    try:
        while True:
            frame = detector.get_frame()
            if frame is None:
                time.sleep(0.02)
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

            cv2.imshow("AutoSort preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.stop()
        cv2.destroyAllWindows()
