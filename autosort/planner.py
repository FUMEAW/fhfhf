"""High-level AutoSort orchestration and sorting plan generation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autosort.config import BIN_ANGLES, DEFAULT_BIN, LABEL_ALIASES, SLOT_MAP, SORTING_RULES
from hardware.platform_control import PlatformController
from hardware.servo_controller import ClosedLoopRotationController
from vision_stuff.vision import DetectionResult, YoloDetector


@dataclass
class HardwareStatus:
    camera_available: bool = False
    camera_index: Optional[int] = None
    encoder_available: bool = False
    platform_available: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class SortAction:
    label: str
    bin_name: str
    slot: int
    door_id: int
    confidence: float
    rotation_deg: float


class BridgeRPC:
    """Thin RPC-like interface between the planner and the hardware."""

    def __init__(self, rotation_controller: ClosedLoopRotationController, platform: PlatformController):
        self.rotation = rotation_controller
        self.platform = platform

    def tilt_to_slot(self, slot: int) -> None:
        self.platform.tilt_to_slot(slot)

    def rotate_to_bin(self, bin_name: str) -> None:
        angle = BIN_ANGLES.get(bin_name.lower(), 0.0)
        self.rotation.move_to(angle)

    def level_platform(self) -> None:
        self.platform.level_platform()

    def open_door(self, door_id: int) -> None:
        self.platform.open_door(door_id)

    def close_door(self, door_id: int) -> None:
        self.platform.close_door(door_id)

    def home(self) -> None:
        self.platform.home()
        self.rotation.move_to(0.0)

    def run_full_sort_sequence(self, action: SortAction) -> None:
        self.rotate_to_bin(action.bin_name)
        self.tilt_to_slot(action.slot)
        self.open_door(action.door_id)
        time.sleep(0.2)
        self.platform.vibrate_slot(action.slot, amplitude_deg=8.0, duration_sec=0.7)
        time.sleep(0.2)
        self.close_door(action.door_id)
        self.level_platform()
        self.home()


class AutoSortSystem:
    """High-level orchestrator for the full sorting cycle."""

    CONFIDENCE_FLOOR = 0.45

    def __init__(self, camera_index: Optional[int] = None, model_path: Optional[str] = None, frame_path: Optional[str] = None):
        self.detector = YoloDetector(camera_index=camera_index, model_path=model_path, conf_threshold=self.CONFIDENCE_FLOOR)
        self.rotation = ClosedLoopRotationController()
        self.platform = PlatformController()
        self.bridge = BridgeRPC(self.rotation, self.platform)
        self.frame_path = Path(frame_path) if frame_path else Path(__file__).resolve().parent.parent / "tmp" / "autosort_frame.jpg"
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        self.hardware_status = HardwareStatus()

    def refresh_hardware_status(self) -> HardwareStatus:
        camera_index = self.detector.camera_index if self.detector.camera_index is not None else self.detector.find_camera_index()
        camera_available = bool(camera_index is not None and self.detector.camera_available)
        encoder_available = bool(getattr(self.rotation, "encoder_available", False) or getattr(self.rotation.servo, "encoder_available", False))
        platform_available = bool(getattr(self.platform.controller, "available", False) or getattr(self.platform.controller, "pwm", None) is not None)

        warnings: list[str] = []
        if not camera_available:
            warnings.append("Camera unavailable")
        if not encoder_available:
            warnings.append("AS5600 encoder unavailable")
        if not platform_available:
            warnings.append("PCA9685 platform controller unavailable")

        self.hardware_status = HardwareStatus(
            camera_available=camera_available,
            camera_index=camera_index,
            encoder_available=encoder_available,
            platform_available=platform_available,
            warnings=warnings,
        )
        return self.hardware_status

    def classify(self) -> Optional[DetectionResult]:
        detections = self.detector.get_detections()
        if not detections:
            return None

        best = max(detections, key=lambda item: item.confidence)
        if best.confidence < self.CONFIDENCE_FLOOR:
            return None
        return best

    def _resolve_bin(self, label: str) -> str:
        key = (label or "").lower().strip()
        if not key:
            return DEFAULT_BIN
        if key in LABEL_ALIASES:
            mapped = LABEL_ALIASES[key]
            if mapped is None:
                return DEFAULT_BIN
            key = mapped
        if key in BIN_ANGLES:
            return key
        if key in SORTING_RULES:
            return SORTING_RULES[key]
        return DEFAULT_BIN

    def _slot_for_bin(self, bin_name: str, slot_count: int = 4) -> int:
        slot = SLOT_MAP.get(bin_name.lower(), 1)
        return max(1, min(slot_count, slot))

    def _door_for_slot(self, slot: int) -> int:
        return slot

    def make_sort_action(self, detection: DetectionResult) -> Optional[SortAction]:
        if detection is None:
            return None

        label = detection.label or "unknown"
        bin_name = self._resolve_bin(label)
        slot = self._slot_for_bin(bin_name)
        door_id = self._door_for_slot(slot)

        return SortAction(
            label=label,
            bin_name=bin_name,
            slot=slot,
            door_id=door_id,
            confidence=detection.confidence,
            rotation_deg=BIN_ANGLES[bin_name],
        )

    def run_sort_cycle(self, detection: Optional[DetectionResult]) -> Optional[SortAction]:
        if detection is None:
            return None

        if not self.refresh_hardware_status().camera_available:
            return None

        action = self.make_sort_action(detection)
        if action is None:
            return None

        if not self.hardware_status.platform_available:
            return None

        self.bridge.run_full_sort_sequence(action)
        return action

    def start(self) -> None:
        try:
            self.detector.start()
        except RuntimeError:
            self.detector._running = False
        self.refresh_hardware_status()
        if self.hardware_status.encoder_available:
            self.rotation.set_zero()
            self.rotation.set_motion_range()
        else:
            self.rotation.current_angle_deg = 0.0
            self.rotation.target_deg = 0.0

    def stop(self) -> None:
        self.detector.stop()
        try:
            self.platform.home()
        except Exception:
            pass

    def run_loop(self, max_cycles: Optional[int] = None, poll_interval: float = 0.25) -> None:
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            self.refresh_hardware_status()
            if not self.hardware_status.camera_available:
                time.sleep(poll_interval)
                cycles += 1
                continue

            detection = self.classify()
            if detection is not None:
                self.run_sort_cycle(detection)
                time.sleep(0.5)
            else:
                time.sleep(poll_interval)
            cycles += 1


__all__ = ["AutoSortSystem", "BridgeRPC", "SortAction"]
