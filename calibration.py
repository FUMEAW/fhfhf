"""Interactive calibration console for AutoSort flap servos and encoder tuning.

This tool is designed to work with the existing AutoSort hardware stack and the
live AI model in vision_stuff/ai_model. It does not invent a separate model or
label set; it reads the same class_names.txt file already used by the vision
pipeline and reuses the existing YoloDetector implementation.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from hardware.servo_controller import ClosedLoopRotationController, ContinuousRotationServoController, ServoController
from vision_stuff.vision import YoloDetector


@dataclass
class FlapServoConfig:
    min_angle: float = -90.0
    max_angle: float = 90.0
    current_angle: float = 0.0


class FlapCalibration:
    """Calibration controller for flap servos and AS5600-based rotation feedback."""

    def __init__(self, servo_bus: int = 1, as5600_bus: int = 4):
        self.servo_bus = servo_bus
        self.as5600_bus = as5600_bus
        self.servo = ServoController(bus=self.servo_bus)
        self.rotation = ClosedLoopRotationController(servo_bus=self.servo_bus, as5600_bus=self.as5600_bus)
        self.continuous_rotation = ContinuousRotationServoController(servo_bus=self.servo_bus, as5600_bus=self.as5600_bus, servo_channel=3)
        self.detector = YoloDetector(camera_index=None)
        self.model_labels = self._load_model_labels()
        self.slot_limits: Dict[int, FlapServoConfig] = {
            slot: FlapServoConfig() for slot in range(0, 16)
        }

    @staticmethod
    def _load_model_labels() -> List[str]:
        labels_path = Path(__file__).resolve().parent / "vision_stuff" / "ai_model" / "class_names.txt"
        if not labels_path.exists():
            return []
        labels = []
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned:
                labels.append(cleaned)
        return labels

    def status(self) -> Dict[str, object]:
        camera_available = self.detector.find_camera_index() is not None
        encoder_available = bool(getattr(self.rotation, "encoder_available", False))
        servo_available = bool(getattr(self.servo, "available", False))
        return {
            "camera_available": camera_available,
            "camera_index": self.detector.find_camera_index(),
            "encoder_available": encoder_available,
            "servo_available": servo_available,
            "model_labels": self.model_labels,
            "slot_limits": {slot: {"min": cfg.min_angle, "max": cfg.max_angle, "current": cfg.current_angle}
                            for slot, cfg in sorted(self.slot_limits.items()) if cfg.max_angle != 90.0 or cfg.min_angle != -90.0 or cfg.current_angle != 0.0},
        }

    def set_limit(self, slot: int, min_angle: float, max_angle: float) -> Dict[str, float]:
        if slot < 0 or slot > 15:
            raise ValueError("slot must be between 0 and 15")
        if max_angle < min_angle:
            raise ValueError("max_angle must be >= min_angle")

        cfg = self.slot_limits.setdefault(slot, FlapServoConfig())
        cfg.min_angle = float(min_angle)
        cfg.max_angle = float(max_angle)
        return {"min": cfg.min_angle, "max": cfg.max_angle}

    def move_servo(self, slot: int, angle: float) -> float:
        if slot < 0 or slot > 15:
            raise ValueError("slot must be between 0 and 15")

        cfg = self.slot_limits.setdefault(slot, FlapServoConfig())
        angle = float(angle)
        bounded = max(cfg.min_angle, min(cfg.max_angle, angle))
        self.servo.set_servo_angle(slot, bounded)
        cfg.current_angle = bounded
        return bounded

    def test_movement(self, slot: int, angle: float, dwell: float = 1.0) -> Dict[str, object]:
        bounded = self.move_servo(slot, angle)
        time.sleep(dwell)
        encoder_angle = self.rotation.read_encoder_angle() if self.rotation.encoder_available else None
        return {
            "slot": slot,
            "angle": bounded,
            "encoder_angle": encoder_angle,
            "encoder_available": bool(self.rotation.encoder_available),
        }

    def set_rotation_zero(self) -> float:
        if not self.rotation.encoder_available:
            raise RuntimeError("AS5600 encoder is not available. Cannot zero rotation.")
        return self.rotation.set_zero()

    def read_encoder(self) -> Optional[float]:
        if not self.rotation.encoder_available:
            return None
        return self.rotation.read_encoder_angle()

    def spin_continuous_servo(self, target_normalized: float = 0.5, duration_sec: float = 3.0) -> Dict[str, object]:
        """Spin the continuous-rotation servo on channel 3 using AS5600 feedback via PID.
        
        Args:
            target_normalized: 0-1 normalized position (0=min 268.33°, 1=max 286.70°)
            duration_sec: how long to run the motion
        """
        if not self.continuous_rotation.servo_available:
            raise RuntimeError("PCA9685 is not available; cannot run the continuous servo.")
        
        # Try to refresh encoder connection if needed
        if not self.continuous_rotation.refresh_encoder():
            raise RuntimeError("AS5600 encoder is not available on bus 4; cannot close the PID loop. Check I2C connections.")
        
        return self.continuous_rotation.spin_to_target(target_normalized=target_normalized, duration_sec=duration_sec)

    def live_ai_snapshot(self, seconds: float = 2.0, print_every: float = 0.25) -> List[dict]:
        """Use the existing YOLO detector and model labels from vision_stuff/ai_model."""
        if self.detector.model is None:
            raise RuntimeError("YOLO model is not available. Ensure ultralytics is installed and the model exists.")

        self.detector.start()
        end_time = time.monotonic() + seconds
        snapshots: List[dict] = []
        try:
            while time.monotonic() < end_time:
                detections = self.detector.get_detections()
                if detections:
                    snapshot = []
                    for detection in detections:
                        snapshot.append(
                            {
                                "label": detection.label,
                                "confidence": round(detection.confidence, 3),
                                "center": detection.center,
                                "slot": detection.slot_num,
                            }
                        )
                    snapshots.append({"time": round(time.monotonic(), 2), "detections": snapshot})
                time.sleep(print_every)
        finally:
            self.detector.stop()

        return snapshots

    def print_ai_snapshot(self, seconds: float = 2.0) -> None:
        try:
            snapshots = self.live_ai_snapshot(seconds=seconds)
        except RuntimeError as exc:
            print(f"AI model unavailable: {exc}")
            return

        if not snapshots:
            print("No detections found in the last %.1f seconds" % seconds)
            return

        print("AI detections:")
        for snapshot in snapshots:
            print(f"  t={snapshot['time']}s")
            for item in snapshot["detections"]:
                print(f"    - {item['label']} conf={item['confidence']} center={item['center']} slot={item['slot']}")


class CalibrationCLI:
    """Command prompt for real-time calibration and testing."""

    HELP_TEXT = """
Commands:
  status
  labels
  set_limit <slot> <min> <max>
  move <slot> <angle>
  test <slot> <angle> [dwell]
  encoder                    - read current encoder angle once
  watch [interval_sec]        - continuously read encoder (default 0.5s, press Ctrl+C to stop)
  zero
  spin <normalized_0_to_1> [seconds]  - spin to normalized position (0=min 268.33°, 1=max 286.70°)
  ai [seconds]
  help
  quit
""".strip()

    def __init__(self):
        self.cal = FlapCalibration()

    def run(self) -> None:
        print("AutoSort calibration console")
        print("Using the live AI model from vision_stuff/ai_model")
        print(self.HELP_TEXT)

        while True:
            try:
                raw = input("cal> ")
            except EOFError:
                print()
                break

            if not raw.strip():
                continue

            try:
                tokens = shlex.split(raw.strip())
            except ValueError as exc:
                print(f"Command parse error: {exc}")
                continue

            cmd = tokens[0].lower()
            args = tokens[1:]

            try:
                if cmd in {"quit", "exit"}:
                    print("Exiting calibration console")
                    break
                if cmd == "help":
                    print(self.HELP_TEXT)
                elif cmd == "status":
                    print(self.cal.status())
                elif cmd == "labels":
                    print("Model labels:")
                    for label in self.cal.model_labels:
                        print(f"  - {label}")
                elif cmd == "set_limit":
                    if len(args) != 3:
                        print("Usage: set_limit <slot> <min> <max>")
                        continue
                    slot = int(args[0])
                    min_angle = float(args[1])
                    max_angle = float(args[2])
                    print(self.cal.set_limit(slot, min_angle, max_angle))
                elif cmd == "move":
                    if len(args) != 2:
                        print("Usage: move <slot> <angle>")
                        continue
                    slot = int(args[0])
                    angle = float(args[1])
                    print(f"Moved slot {slot} to angle {self.cal.move_servo(slot, angle)}")
                elif cmd == "test":
                    if len(args) not in {2, 3}:
                        print("Usage: test <slot> <angle> [dwell]")
                        continue
                    slot = int(args[0])
                    angle = float(args[1])
                    dwell = float(args[2]) if len(args) == 3 else 1.0
                    print(self.cal.test_movement(slot, angle, dwell=dwell))
                elif cmd == "encoder":
                    angle = self.cal.read_encoder()
                    if angle is None:
                        print("AS5600 encoder unavailable")
                    else:
                        print(f"Encoder angle: {angle:.2f} deg")
                elif cmd == "watch":
                    interval = float(args[0]) if args else 0.5
                    interval = max(0.01, min(interval, 10.0))
                    print(f"Watching encoder at {interval}s intervals. Press Ctrl+C to stop.")
                    try:
                        while True:
                            angle = self.cal.read_encoder()
                            if angle is None:
                                print("\r[unavailable]", end="", flush=True)
                            else:
                                print(f"\rEncoder: {angle:7.2f}°", end="", flush=True)
                            time.sleep(interval)
                    except KeyboardInterrupt:
                        print("\nWatch stopped.")
                elif cmd == "zero":
                    print(f"Zeroed encoder: {self.cal.set_rotation_zero()}")
                elif cmd == "spin":
                    target_norm = float(args[0]) if args else 0.5
                    duration = float(args[1]) if len(args) > 1 else 3.0
                    target_norm = max(0.0, min(1.0, target_norm))
                    print(self.cal.spin_continuous_servo(target_normalized=target_norm, duration_sec=duration))
                elif cmd == "ai":
                    seconds = float(args[0]) if args else 2.0
                    self.cal.print_ai_snapshot(seconds=seconds)
                else:
                    print(f"Unknown command: {cmd}")
                    print(self.HELP_TEXT)
            except Exception as exc:  # pragma: no cover - user-facing CLI safety net.
                print(f"Error: {exc}")


if __name__ == "__main__":
    CalibrationCLI().run()
