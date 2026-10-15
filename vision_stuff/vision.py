"""Vision pipeline for the AutoSort system.

Uses onnxruntime InferenceSession directly (session pattern borrowed from
daslearning-org/vision-ai onnx_detect.py), no ultralytics dependency.

The exported model in vision_stuff/ai_model is a Roboflow RF-DETR (small)
detector, NOT a YOLO model. That matters for both ends of the pipeline:

Preprocessing (from inference_config.json):
  - Stretch-resize to the model input size (512x512)
  - BGR -> RGB, scale to [0, 1]
  - ImageNet mean/std normalization
  - HWC -> CHW, add batch dimension

Outputs:
  - "dets":   (1, 300, 4)  boxes as normalized (cx, cy, w, h) in [0, 1]
  - "labels": (1, 300, C)  raw class logits -> sigmoid (NOT softmax)
  Class index i maps to line i of class_names.txt; index 0 is a background
  placeholder and is ignored. DETR models are set-based, so no NMS is needed;
  each query keeps its single best non-background class.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from autosort.config import LABEL_ALIASES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in original pixels
    center: Tuple[int, int]
    class_id: int
    slot_num: Optional[int] = None


@dataclass
class VisionStats:
    """Telemetry snapshot of the inference thread."""
    session_ok: bool = False
    camera_available: bool = False
    inference_count: int = 0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    last_inference_time: float = 0.0
    # time.time() when the frame behind the current detections was captured
    last_frame_time: float = 0.0
    frame_shape: Optional[Tuple[int, int]] = None
    # Highest score seen per class on the last frame, even below threshold.
    class_peaks: Dict[str, float] = field(default_factory=dict)
    last_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Slot geometry
# ---------------------------------------------------------------------------

# Camera at the landfill corner looking across the platform. The X arms run
# diagonally through the hub, so each slot is a wedge: top, right, bottom, left.
DEFAULT_SLOT_LAYOUT = {
    "center": [0.51, 0.625],  # platform hub as a fraction of image width/height
    "slots": {"top": 2, "right": 4, "bottom": 3, "left": 1},
}


def wedge_for_point(x: float, y: float, width: int, height: int, center=(0.5, 0.5)) -> Optional[str]:
    """Which diagonal wedge around the hub a point is in."""
    if width <= 0 or height <= 0:
        return None
    dx = x - center[0] * width
    dy = y - center[1] * height
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


def slot_for_point(x: float, y: float, width: int, height: int, layout: Optional[dict] = None) -> Optional[int]:
    layout = layout or DEFAULT_SLOT_LAYOUT
    wedge = wedge_for_point(x, y, width, height, layout["center"])
    return None if wedge is None else int(layout["slots"][wedge])


def draw_slot_wedges(img: np.ndarray, layout: Optional[dict] = None) -> None:
    """Draw the hub, the diagonal slot boundaries, and each wedge's slot number."""
    layout = layout or DEFAULT_SLOT_LAYOUT
    h, w = img.shape[:2]
    cx, cy = int(layout["center"][0] * w), int(layout["center"][1] * h)
    reach = max(w, h)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * reach, cy + sy * reach), (255, 255, 0), 1)
    cv2.circle(img, (cx, cy), 4, (255, 255, 0), -1)
    offset = int(0.22 * min(w, h))
    spots = {"top": (0, -offset), "right": (offset, 0), "bottom": (0, offset), "left": (-offset, 0)}
    for wedge, (ox, oy) in spots.items():
        cv2.putText(img, f"S{layout['slots'][wedge]}", (cx + ox - 12, cy + oy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YoloDetector:
    """Threaded ONNX RF-DETR detector for AutoSort.

    The class name is kept for backwards compatibility with the planner.

    Usage:
        detector = YoloDetector(camera_index=2, conf_threshold=0.45)
        detector.start()          # opens camera + inference threads
        detections = detector.get_detections()
        frame = detector.get_frame()
        stats = detector.get_stats()
        detector.stop()
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        input_size: Optional[int] = None,
        num_threads: int = 4,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "vision_stuff" / "ai_model"

        # Search for an existing model file in order of preference
        default_model = model_dir / "weights.onnx"
        for candidate in ("weights.onnx", "best.onnx", "model.onnx"):
            p = model_dir / candidate
            if p.exists():
                default_model = p
                break
        self.model_path = Path(model_path) if model_path else default_model

        # class_names.txt line i == model class index i (index 0 = background)
        class_names_path = model_dir / "class_names.txt"
        self._class_names: List[str] = []
        if class_names_path.exists():
            self._class_names = [
                line.strip() for line in class_names_path.read_text().strip().splitlines()
            ]

        # Normalization and input size from the Roboflow inference config
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self._std = np.array(IMAGENET_STD, dtype=np.float32)
        cfg_size: Optional[int] = None
        cfg_path = model_dir / "inference_config.json"
        if cfg_path.exists():
            try:
                net = json.loads(cfg_path.read_text()).get("network_input", {})
                norm = net.get("normalization")
                if norm and len(norm) == 2:
                    self._mean = np.array(norm[0], dtype=np.float32)
                    self._std = np.array(norm[1], dtype=np.float32)
                cfg_size = int(net.get("training_input_size", {}).get("width", 0)) or None
            except Exception as exc:
                print(f"WARNING: could not parse {cfg_path.name}: {exc}")

        self.camera_index = camera_index
        self.conf_threshold = float(conf_threshold)
        self.input_size = int(input_size or cfg_size or 512)
        self.num_threads = int(num_threads)
        self.camera_available = False
        # Image -> slot mapping; the planner overrides this from calibration.
        self.slot_layout = {
            "center": list(DEFAULT_SLOT_LAYOUT["center"]),
            "slots": dict(DEFAULT_SLOT_LAYOUT["slots"]),
        }

        # Session state, populated by start_session()
        self._sess: Optional[InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._boxes_idx: int = 0
        self._logits_idx: int = 1

        # Camera + threading state
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._frame_lock = threading.Lock()

        self._detections: List[DetectionResult] = []
        self._detections_frame_time: float = 0.0
        self._detections_lock = threading.Lock()

        self._stats = VisionStats()
        self._stats_lock = threading.Lock()

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Session management (mirrors start_detect_session() from reference)
    # ------------------------------------------------------------------

    def start_session(self) -> bool:
        """Load the ONNX model and start an InferenceSession.

        Returns True on success. Safe to call multiple times.
        """
        if self._sess is not None:
            return True

        if not self.model_path.exists():
            print(f"ONNX model not found: {self.model_path}")
            return False

        try:
            opts = SessionOptions()
            opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = self.num_threads
            self._sess = InferenceSession(
                str(self.model_path), opts, providers=["CPUExecutionProvider"]
            )
            inp = self._sess.get_inputs()[0]
            self._input_name = inp.name
            # Trust the model's fixed input size over any config default
            if len(inp.shape) == 4 and isinstance(inp.shape[2], int):
                self.input_size = int(inp.shape[2])

            outputs = self._sess.get_outputs()
            self._output_names = [o.name for o in outputs]
            # Identify the box tensor (last dim 4) vs. the logits tensor
            for i, o in enumerate(outputs):
                if o.name == "dets" or (o.shape and o.shape[-1] == 4):
                    self._boxes_idx = i
                    self._logits_idx = 1 - i if len(outputs) == 2 else (i + 1) % len(outputs)
                    break
            num_classes = outputs[self._logits_idx].shape[-1]
            if self._class_names and num_classes != len(self._class_names):
                print(
                    f"WARNING: model has {num_classes} classes but class_names.txt "
                    f"has {len(self._class_names)} entries"
                )
            print(
                f"ONNX session started: {self.model_path.name} | "
                f"input={self._input_name} {inp.shape} | "
                f"outputs={[(o.name, o.shape) for o in outputs]} | "
                f"classes={self._class_names}"
            )
            with self._stats_lock:
                self._stats.session_ok = True
            return True
        except Exception as exc:
            print(f"Error loading ONNX model: {exc}")
            self._sess = None
            with self._stats_lock:
                self._stats.last_error = f"model load: {exc}"
            return False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> ImageNet-normalized float32 (1, 3, H, W) tensor."""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        chw = rgb.transpose(2, 0, 1)
        return np.ascontiguousarray(chw[np.newaxis], dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run one inference pass on a BGR frame. Returns DetectionResult list."""
        if self._sess is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        img_data = self._preprocess(frame)

        t0 = time.perf_counter()
        try:
            results = self._sess.run(self._output_names, {self._input_name: img_data})
        except Exception as exc:
            print(f"Inference error: {exc}")
            with self._stats_lock:
                self._stats.last_error = f"inference: {exc}"
            return []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = np.asarray(results[self._boxes_idx])[0]      # (Q, 4)
        logits = np.asarray(results[self._logits_idx])[0]    # (Q, C)
        detections, peaks = self._parse_output(boxes, logits, orig_w, orig_h)

        with self._stats_lock:
            s = self._stats
            s.inference_count += 1
            s.last_inference_ms = elapsed_ms
            s.avg_inference_ms = (
                elapsed_ms if s.inference_count == 1
                else 0.9 * s.avg_inference_ms + 0.1 * elapsed_ms
            )
            s.last_inference_time = time.time()
            s.frame_shape = (orig_w, orig_h)
            s.class_peaks = peaks
            s.last_error = None

        return detections

    # Backwards-compatible alias
    _run_inference = run_inference

    # ------------------------------------------------------------------
    # Output parsing (RF-DETR / DETR style)
    # ------------------------------------------------------------------

    def _class_name(self, class_id: int) -> str:
        if self._class_names and class_id < len(self._class_names):
            return self._class_names[class_id]
        return str(class_id)

    def _parse_output(
        self, boxes: np.ndarray, logits: np.ndarray, orig_w: int, orig_h: int
    ) -> Tuple[List[DetectionResult], Dict[str, float]]:
        """Convert raw RF-DETR outputs into detections in original pixel coords."""
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))  # sigmoid
        scores[:, 0] = 0.0  # index 0 is the background placeholder

        peaks = {
            self._class_name(c): float(scores[:, c].max())
            for c in range(1, scores.shape[1])
        }

        best_class = scores.argmax(axis=1)
        best_score = scores[np.arange(len(scores)), best_class]
        keep = np.where(best_score >= self.conf_threshold)[0]
        keep = keep[np.argsort(-best_score[keep])]

        detections: List[DetectionResult] = []
        for q in keep:
            cx, cy, bw, bh = (float(v) for v in boxes[q])
            x1 = max(0.0, (cx - bw / 2) * orig_w)
            y1 = max(0.0, (cy - bh / 2) * orig_h)
            x2 = min(float(orig_w), (cx + bw / 2) * orig_w)
            y2 = min(float(orig_h), (cy + bh / 2) * orig_h)
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(best_class[q])
            normalized = self._class_name(class_id).lower().strip()
            mapped = LABEL_ALIASES.get(normalized, normalized)
            if not mapped:
                continue

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            detections.append(DetectionResult(
                label=mapped,
                class_id=class_id,
                confidence=float(best_score[q]),
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                slot_num=self.get_slot(center_x, center_y, orig_w, orig_h),
            ))

        return detections, peaks

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

    def get_slot(self, center_x: float, center_y: float, width: int = 1, height: int = 1) -> Optional[int]:
        """Slot (1-4) for an image point, using this detector's slot_layout."""
        return slot_for_point(center_x, center_y, width, height, self.slot_layout)

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

    def get_stats(self) -> VisionStats:
        with self._stats_lock:
            snap = VisionStats(**{k: getattr(self._stats, k) for k in self._stats.__dataclass_fields__})
            snap.class_peaks = dict(self._stats.class_peaks)
        snap.camera_available = self.camera_available
        return snap

    def get_detections_since(self, since: float) -> Optional[List[DetectionResult]]:
        """Detections from a frame captured after `since` (time.time()), else None."""
        with self._detections_lock:
            if self._detections_frame_time <= since:
                return None
            return list(self._detections)

    def wait_for_detections_since(
        self, since: float, timeout: float = 60.0, poll: float = 0.1
    ) -> Optional[List[DetectionResult]]:
        """Block until a frame captured after `since` has been analyzed."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            dets = self.get_detections_since(since)
            if dets is not None:
                return dets
            if not self._running:
                return None
            time.sleep(poll)
        return None

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
                self._frame_time = time.time()

    def _inference_loop(self) -> None:
        while self._running:
            with self._frame_lock:
                frame = None if self._frame is None else self._frame.copy()
                frame_time = self._frame_time

            if frame is None:
                time.sleep(0.01)
                continue

            detections = self.run_inference(frame)

            with self._detections_lock:
                self._detections = detections
                self._detections_frame_time = frame_time
            with self._stats_lock:
                self._stats.last_frame_time = frame_time

            if self._sess is None:
                time.sleep(0.5)


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
