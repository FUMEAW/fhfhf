"""High-level AutoSort orchestration and sorting plan generation.

Sort flow for each item on the platform:
    find the slot the item sits in (camera wedge) -> pick its bin (AI label) ->
    turn the platform so that slot is over that bin -> tilt toward the slot ->
    open the slot's flap -> vibrate -> close -> level -> turn back home

Bins are fixed at the corners and slots ride on the platform (see
"Platform Drawing.png"), so the turn is autosort.config.turn_for(slot, bin):
0, 90, 180 or 270 degrees. Each turn has a calibrated yaw position (0-1 on the
AS5600 encoder arc), driven by the PID loop in ContinuousRotationServoController.
All mechanical values come from hardware/calibration.json (edit them live with
calibrate_platform.py).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from autosort.config import (
    BIN_ANGLES,
    BIN_DIRECTIONS,
    DEFAULT_BIN,
    LABEL_ALIASES,
    SORTING_RULES,
    home_slot_for_bin,
    turn_for,
)
from hardware.calibration_store import DEFAULT_TURN_POSITIONS, load_calibration
from hardware.platform_control import PlatformController
from hardware.servo_controller import ClosedLoopRotationController
from vision_stuff.vision import DetectionResult, YoloDetector, draw_slot_wedges


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
    turn: int            # clockwise platform turn in degrees: 0/90/180/270
    rotation_pos: float  # yaw target for that turn, 0-1 on the encoder arc
    center: Optional[tuple] = None

    def describe(self) -> str:
        return (
            f"{self.label} ({self.confidence:.2f}) in slot {self.slot} -> "
            f"bin {self.bin_name}: turn {self.turn} deg (yaw {self.rotation_pos:.3f})"
        )


def apply_pid_tuning(rotation: ClosedLoopRotationController, pid: Optional[Dict[str, float]]) -> None:
    """Push calibrated PID gains into the yaw controller, if any were saved."""
    if not pid:
        return
    cont = getattr(rotation, "cont", None)
    if cont is not None and getattr(cont, "pid", None) is not None:
        kp, ki, kd = cont.pid.tunings
        cont.pid.tunings = (pid.get("kp", kp), pid.get("ki", ki), pid.get("kd", kd))


class RotationError(RuntimeError):
    """The platform did not reach the turn needed for a drop."""


class BridgeRPC:
    # How far (encoder degrees) the yaw may end from its target and still drop.
    # move_to_normalized() itself reports success only within 1 degree.
    DROP_TOLERANCE_DEG = 3.0

    def __init__(
        self,
        rotation_controller: ClosedLoopRotationController,
        platform: PlatformController,
        turn_positions: Optional[Dict[int, float]] = None,
        rotation_timeout: float = 15.0,
    ):
        self.rotation = rotation_controller
        self.platform = platform
        self.turn_positions = {
            int(k): float(v)
            for k, v in (turn_positions if turn_positions is not None else DEFAULT_TURN_POSITIONS).items()
        }
        self.rotation_timeout = float(rotation_timeout)

    @property
    def rotation_home(self) -> float:
        return self.turn_positions[0]

    def rotate_to_position(self, position: float) -> bool:
        """PID-drive the yaw servo to a 0-1 position. Returns True if it arrived."""
        position = max(0.0, min(1.0, float(position)))
        if not getattr(self.rotation, "encoder_available", False) and not self.rotation._refresh():
            print(f"[sort] yaw skipped: AS5600 encoder unavailable (target {position:.3f})")
            return False
        ok = self.rotation.move_to_normalized(position, timeout=self.rotation_timeout)
        if ok:
            return True
        if not getattr(self.rotation, "encoder_available", False):
            print(f"[sort] yaw stopped: lost the AS5600 encoder (target {position:.3f})")
            return False
        miss = abs(float(self.rotation.current_angle_deg) - float(self.rotation.target_deg))
        if miss <= self.DROP_TOLERANCE_DEG:
            return True
        print(
            f"[sort] yaw did not settle: target {position:.3f} "
            f"({self.rotation.target_deg:.1f} deg), ended at "
            f"{self.rotation.current_angle_deg:.1f} deg, {miss:.1f} deg off"
        )
        return False

    def rotate_to_turn(self, turn: int) -> bool:
        return self.rotate_to_position(self.turn_positions[int(turn) % 360])

    def rotate_slot_to_bin(self, slot: int, bin_name: str) -> bool:
        return self.rotate_to_turn(turn_for(slot, bin_name))

    def tilt_to_slot(self, slot: int) -> None:
        self.platform.tilt_to_slot(slot)

    def level_platform(self) -> None:
        self.platform.level_platform()

    def open_door(self, door_id: int) -> None:
        self.platform.open_door(door_id)

    def close_door(self, door_id: int) -> None:
        self.platform.close_door(door_id)

    def home(self) -> None:
        self.platform.home()
        self.rotate_to_turn(0)

    def run_full_sort_sequence(self, action: SortAction) -> None:
        try:
            # Never open a flap unless the platform is confirmed over the right bin.
            if not self.rotate_to_turn(action.turn):
                raise RotationError(
                    f"platform did not reach turn {action.turn} for {action.bin_name}; "
                    f"drop aborted, item left in slot {action.slot}"
                )
            # Tilting opens this slot's flap and keeps every other flap closed.
            self.tilt_to_slot(action.slot)
            time.sleep(0.2)
            self.platform.vibrate_slot(action.slot, amplitude_deg=8.0, duration_sec=0.7)
            time.sleep(0.2)
            self.close_door(action.door_id)
            self.level_platform()
        finally:
            # Always return to a safe rest state, even if a step failed.
            self.rotation.stop()
            self.home()


class AutoSortSystem:
    # Raised from 0.45: the model scores an EMPTY platform as "recycle" ~0.6,
    # while real items score 0.94-0.98. Override with --conf in main.py.
    CONFIDENCE_FLOOR = 0.7
    # Ignore frames captured within this long after the platform stops moving.
    SETTLE_SEC = 0.75
    # After a sort sequence fails (e.g. yaw could not reach its turn), wait this
    # long before trying again so a dead encoder doesn't cause constant retries.
    RETRY_COOLDOWN_SEC = 15.0

    def __init__(
        self,
        camera_index: Optional[int] = None,
        model_path: Optional[str] = None,
        frame_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        dry_run: bool = False,
    ):
        if conf_threshold is not None:
            self.CONFIDENCE_FLOOR = float(conf_threshold)
        self.dry_run = bool(dry_run)

        self.detector = YoloDetector(
            camera_index=camera_index,
            model_path=model_path,
            conf_threshold=self.CONFIDENCE_FLOOR,
        )

        self.calibration = load_calibration()
        self.rotation = ClosedLoopRotationController()
        apply_pid_tuning(self.rotation, self.calibration.get("pid"))
        self.platform = PlatformController(calibration=self.calibration)
        self.bridge = BridgeRPC(
            self.rotation,
            self.platform,
            turn_positions=self.calibration["turn_positions"],
        )
        self.detector.slot_layout = self.calibration["camera"]
        if not self.calibration.get("turns_calibrated"):
            print(
                "WARNING: yaw turn positions are not calibrated; using placeholder guesses "
                f"{self.bridge.turn_positions}. Run: python calibrate_platform.py"
            )

        self.tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.frame_path = (
            Path(frame_path) if frame_path else self.tmp_dir / "autosort_frame.jpg"
        )
        self.detection_json_path = self.tmp_dir / "latest_detection.json"
        self.history_path = self.tmp_dir / "sort_history.jsonl"
        self.hardware_status = HardwareStatus()
        self.state = "starting"
        self.sorted_count = 0
        self.last_action: Optional[SortAction] = None
        self._last_motion_end = 0.0
        self.failed_count = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def refresh_hardware_status(self) -> HardwareStatus:
        camera_index = self.detector.camera_index
        camera_available = bool(self.detector.camera_available)
        encoder_available = bool(
            getattr(self.rotation, "encoder_available", False)
            or getattr(self.rotation.servo, "encoder_available", False)
        )
        platform_available = bool(
            getattr(self.platform.controller, "available", False)
            or getattr(self.platform.controller, "pwm", None) is not None
        )

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

    # ------------------------------------------------------------------
    # Detection -> plan
    # ------------------------------------------------------------------

    def classify(self) -> Optional[DetectionResult]:
        detections = self.detector.get_detections()
        if not detections:
            return None
        best = max(detections, key=lambda item: item.confidence)
        if best.confidence < self.CONFIDENCE_FLOOR:
            return None
        return best

    def _publish_frame_and_detections(self) -> Optional[DetectionResult]:
        """Pull frame from vision detector, draw bounding boxes, and save for Streamlit."""
        frame = self.detector.get_frame()
        detections = self.detector.get_detections()

        if frame is not None:
            annotated = frame.copy()
            self._draw_slot_wedges(annotated)
            for det in detections:
                if det.bbox:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    color = (0, 255, 0) if det.confidence >= self.CONFIDENCE_FLOOR else (0, 200, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(annotated, det.center, 5, (0, 0, 255), -1)
                    label_text = f"{det.label} {det.confidence:.2f} s{det.slot_num}"
                    cv2.putText(
                        annotated,
                        label_text,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
            # Write to a temp file then rename, so the dashboard never reads a half-written image
            try:
                tmp_path = self.frame_path.with_name(self.frame_path.stem + ".tmp.jpg")
                if cv2.imwrite(str(tmp_path), annotated):
                    os.replace(tmp_path, self.frame_path)
            except Exception:
                pass

        best_detection = None
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            if best.confidence >= self.CONFIDENCE_FLOOR:
                best_detection = best

        self.write_status(detections, best_detection)
        return best_detection

    def _draw_slot_wedges(self, img) -> None:
        draw_slot_wedges(img, self.detector.slot_layout)

    def write_status(
        self,
        detections: Optional[List[DetectionResult]] = None,
        best_detection: Optional[DetectionResult] = None,
    ) -> None:
        """Write tmp/latest_detection.json, the dashboard's single source of truth."""
        if detections is None:
            detections = self.detector.get_detections()
        stats = self.detector.get_stats() if hasattr(self.detector, "get_stats") else None
        hw = self.hardware_status
        yaw_pos = None
        cont = getattr(self.rotation, "cont", None)
        if cont is not None and hw.encoder_available and self.state != "sorting":
            try:
                yaw_pos = round(cont.degrees_to_normalized(cont.read_encoder_angle()), 4)
            except Exception:
                yaw_pos = None
        record = {
            # Keys the original dashboard already used
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "class": best_detection.label if best_detection else None,
            "confidence": float(best_detection.confidence) if best_detection else 0.0,
            "bin": self._resolve_bin(best_detection.label) if best_detection else None,
            # Extended status
            "heartbeat": time.time(),
            "state": self.state,
            "dry_run": self.dry_run,
            "slot": best_detection.slot_num if best_detection else None,
            "threshold": self.CONFIDENCE_FLOOR,
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(float(d.confidence), 4),
                    "slot": d.slot_num,
                    "bin": self._resolve_bin(d.label),
                    "bbox": [round(float(v), 1) for v in d.bbox],
                }
                for d in detections
            ],
            "vision": None if stats is None else {
                "model_ok": stats.session_ok,
                "inference_count": stats.inference_count,
                "last_inference_ms": round(stats.last_inference_ms, 1),
                "avg_inference_ms": round(stats.avg_inference_ms, 1),
                "class_peaks": {k: round(v, 4) for k, v in stats.class_peaks.items()},
                "error": stats.last_error,
            },
            "hardware": {
                "camera": hw.camera_available,
                "camera_index": hw.camera_index,
                "encoder": hw.encoder_available,
                "platform": hw.platform_available,
                "warnings": hw.warnings,
            },
            "yaw_pos": yaw_pos,
            "turn_positions": {str(k): v for k, v in self.bridge.turn_positions.items()},
            "turns_calibrated": bool(self.calibration.get("turns_calibrated")),
            "sorted_count": self.sorted_count,
            "failed_count": self.failed_count,
            "last_action": self.last_action.describe() if self.last_action else None,
        }
        try:
            tmp_json = self.detection_json_path.with_suffix(".json.tmp")
            with open(tmp_json, "w") as f:
                json.dump(record, f)
            os.replace(tmp_json, self.detection_json_path)
        except Exception:
            pass

    def _record_history(self, action: SortAction, ok: bool, duration: float) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time": time.time(),
            "class": action.label,
            "confidence": round(float(action.confidence), 4),
            "bin": action.bin_name,
            "slot": action.slot,
            "turn": action.turn,
            "yaw_pos": action.rotation_pos,
            "ok": ok,
            "dry_run": self.dry_run,
            "duration_s": round(duration, 2),
        }
        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

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
        """Slot over this bin when the platform is at home (turn 0)."""
        return home_slot_for_bin(bin_name)

    def _door_for_slot(self, slot: int) -> int:
        return slot

    def make_sort_action(self, detection: DetectionResult) -> Optional[SortAction]:
        if detection is None:
            return None
        label = detection.label or "unknown"
        bin_name = self._resolve_bin(label)
        if bin_name not in BIN_DIRECTIONS:
            bin_name = DEFAULT_BIN
        # Dump through the slot the item physically sits in (camera wedge).
        slot = detection.slot_num if detection.slot_num in (1, 2, 3, 4) else home_slot_for_bin(bin_name)
        turn = turn_for(slot, bin_name)
        return SortAction(
            label=label,
            bin_name=bin_name,
            slot=slot,
            door_id=self._door_for_slot(slot),
            confidence=detection.confidence,
            turn=turn,
            rotation_pos=self.bridge.turn_positions[turn],
            center=detection.center,
        )

    def plan_actions(self, detections: List[DetectionResult]) -> List[SortAction]:
        """One action per occupied slot, highest confidence first.

        Two boxes in the same slot share one flap, so only the most confident
        one decides that slot's bin.
        """
        best_per_slot: Dict[int, SortAction] = {}
        for det in detections:
            if det.confidence < self.CONFIDENCE_FLOOR:
                continue
            action = self.make_sort_action(det)
            if action is None:
                continue
            current = best_per_slot.get(action.slot)
            if current is None or action.confidence > current.confidence:
                best_per_slot[action.slot] = action
        return sorted(best_per_slot.values(), key=lambda a: -a.confidence)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_action(self, action: SortAction) -> bool:
        if self.dry_run:
            print(f"[sort] DRY RUN, would sort: {action.describe()}")
            self._record_history(action, ok=False, duration=0.0)
            return False
        self.refresh_hardware_status()
        if not self.hardware_status.platform_available:
            print(f"[sort] skipped, PCA9685 unavailable: {action.describe()}")
            return False
        print(f"[sort] sorting: {action.describe()}")
        self.state = "sorting"
        self.write_status()
        t0 = time.time()
        ok = True
        try:
            self.bridge.run_full_sort_sequence(action)
        except Exception as exc:
            ok = False
            print(f"[sort] FAILED: {exc}")
        finally:
            self.state = "dry-run" if self.dry_run else "running"
        duration = time.time() - t0
        self._record_history(action, ok=ok, duration=duration)
        if not ok:
            self.failed_count += 1
            return False
        self.sorted_count += 1
        self.last_action = action
        print(f"[sort] done in {duration:.1f}s (total sorted: {self.sorted_count})")
        self.write_status()
        return True

    def run_sort_cycle(self, detection: Optional[DetectionResult]) -> Optional[SortAction]:
        """Sort a single detection (kept for backwards compatibility)."""
        action = self.make_sort_action(detection) if detection is not None else None
        if action is None:
            return None
        return action if self.execute_action(action) else None

    def start(self) -> None:
        try:
            self.detector.start()
        except RuntimeError:
            pass
        self.refresh_hardware_status()
        if self.hardware_status.encoder_available:
            self.rotation.set_zero()
            self.rotation.set_motion_range()  # full logical 0-270 degree arc
        else:
            self.rotation.current_angle_deg = 0.0
            self.rotation.target_deg = 0.0
        if not self.dry_run and self.hardware_status.platform_available:
            print("[sort] moving platform to home position")
            try:
                self.bridge.home()
            except Exception as exc:
                print(f"[sort] homing failed: {exc}")
        self._last_motion_end = time.time()
        self.state = "dry-run" if self.dry_run else "running"

    def stop(self) -> None:
        self.state = "stopped"
        try:
            self.write_status([])
        except Exception:
            pass
        self.detector.stop()
        try:
            self.rotation.stop()
        except Exception:
            pass
        if self.dry_run:
            return
        try:
            self.platform.home()
        except Exception:
            pass

    def run_loop(self, max_cycles: Optional[int] = None, poll_interval: float = 0.1) -> None:
        """Main loop. A cycle is one fresh inference result after the platform settles."""
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            self.refresh_hardware_status()
            if not self.detector.camera_available:
                self.write_status([])
                time.sleep(max(poll_interval, 0.5))
                cycles += 1
                continue

            self._publish_frame_and_detections()

            # Only act on frames captured after the platform finished moving,
            # so an item that was just sorted is never sorted twice.
            detections = self.detector.get_detections_since(self._last_motion_end + self.SETTLE_SEC)
            if detections is None:
                time.sleep(poll_interval)
                continue
            cycles += 1

            actions = self.plan_actions(detections)
            if not actions:
                # Consume this result; wait for the next frame.
                self._last_motion_end = time.time() - self.SETTLE_SEC
                time.sleep(poll_interval)
                continue

            if len(actions) > 1:
                print(f"[sort] {len(actions)} items on platform, sorting one by one")
            failed_before = self.failed_count
            for action in actions:
                self.execute_action(action)
            self._last_motion_end = time.time()
            if self.failed_count > failed_before:
                print(f"[sort] sort failed; retrying in {self.RETRY_COOLDOWN_SEC:.0f}s")
                self._last_motion_end += self.RETRY_COOLDOWN_SEC


__all__ = ["AutoSortSystem", "BridgeRPC", "SortAction", "apply_pid_tuning"]
