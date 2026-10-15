"""Platform motion helpers for the AutoSort mechanism.

This module keeps the Stewart-platform orientation, door actuation, and platform
leveling in one place with explicit, well-named operations.
"""

from __future__ import annotations

import time
from typing import Optional

from hardware.calibration_store import (
    DEFAULT_FLAPS,
    DEFAULT_LEVEL_TILT,
    DEFAULT_SLOT_TILT,
    load_calibration,
)
from hardware.config import SLOT_FLAP_CHANNELS, TILT_CHANNELS
from hardware.servo_controller import ServoController


class PlatformController:
    """Drive tilt, door, and stable level logic for the sorting platform.

    PCA9685 channels 0-2 are tilt, channel 3 is the continuous-rotation
    yaw servo (owned by ClosedLoopRotationController). Slot doors are the
    flap servos on channels 4-7, never write door_id as a raw channel.

    Angles come from hardware/calibration.json (see calibrate_platform.py),
    falling back to the defaults in hardware/calibration_store.py.
    """

    # Class-level defaults, kept for reference and backwards compatibility.
    # Instances use self.SLOT_ANGLES / self.FLAP_ANGLES loaded from calibration.
    SLOT_ANGLES = {slot: list(a) for slot, a in DEFAULT_SLOT_TILT.items()}
    SLOT_FLAP_MAP = SLOT_FLAP_CHANNELS
    # Keyed by PCA9685 channel: [open, closed]
    FLAP_ANGLES = {
        SLOT_FLAP_CHANNELS[slot]: [f["open"], f["closed"]] for slot, f in DEFAULT_FLAPS.items()
    }
    LEVEL_ANGLES = list(DEFAULT_LEVEL_TILT)

    def __init__(
        self,
        servo_controller: Optional[ServoController] = None,
        calibration: Optional[dict] = None,
    ):
        self.controller = servo_controller or ServoController()
        self.current_slot = None
        self.current_door_state = {1: "closed", 2: "closed", 3: "closed", 4: "closed"}
        self.apply_calibration(calibration if calibration is not None else load_calibration())

    def apply_calibration(self, cal: dict) -> None:
        """Replace this instance's angles with a calibration dict."""
        self.LEVEL_ANGLES = [float(v) for v in cal["level_tilt"]]
        self.SLOT_ANGLES = {int(s): [float(v) for v in a] for s, a in cal["slot_tilt"].items()}
        self.FLAP_ANGLES = {
            self.SLOT_FLAP_MAP[int(s)]: [float(f["open"]), float(f["closed"])]
            for s, f in cal["flaps"].items()
        }

    def tilt_to_slot(self, slot: int) -> None:
        if slot not in self.SLOT_ANGLES:
            raise ValueError(f"Invalid slot: {slot}. Must be between 1 and 4.")

        angles = self.SLOT_ANGLES[slot]
        for servo_index, angle in zip(TILT_CHANNELS, angles):
            self.controller.set_servo_angle(servo_index, angle)

        # Only the target slot's flap may be open; every other flap stays closed.
        for other in self.SLOT_FLAP_MAP:
            if other != slot:
                self.close_door(other)
        self.open_door(slot)
        self.current_slot = slot

    def level_platform(self) -> None:
        for servo_index, angle in zip(TILT_CHANNELS, self.LEVEL_ANGLES):
            self.controller.set_servo_angle(servo_index, angle)

        for door_id in self.SLOT_FLAP_MAP:
            self.close_door(door_id)
        self.current_slot = None

    def _flap_channel(self, slot: int) -> int:
        if slot not in self.SLOT_FLAP_MAP:
            raise ValueError(f"Invalid door/slot: {slot}. Must be between 1 and 4.")
        return self.SLOT_FLAP_MAP[slot]

    def open_door(self, door_id: int) -> None:
        flap_num = self._flap_channel(door_id)
        self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][0])
        self.current_door_state[door_id] = "open"

    def close_door(self, door_id: int) -> None:
        flap_num = self._flap_channel(door_id)
        self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][1])
        self.current_door_state[door_id] = "closed"

    def vibrate_slot(self, slot: int, amplitude_deg: float = 8.0, duration_sec: float = 0.8) -> None:
        if slot not in self.SLOT_FLAP_MAP:
            raise ValueError(f"Invalid slot: {slot}. Must be between 1 and 4.")

        flap_num = self.SLOT_FLAP_MAP[slot]
        self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][0])

        vibrate = getattr(self.controller, "vibrate_servos_aggressive", None)
        if callable(vibrate):
            vibrate([flap_num], amplitude_deg=amplitude_deg, duration_sec=duration_sec)
        else:
            self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][0])

    def expel_item(self, slot: int, amplitude_deg: float = 8.0, duration_sec: float = 0.8) -> None:
        self.tilt_to_slot(slot)
        self.open_door(slot)
        time.sleep(0.2)
        self.vibrate_slot(slot, amplitude_deg=amplitude_deg, duration_sec=duration_sec)
        time.sleep(0.2)
        self.close_door(slot)
        self.level_platform()

    def home(self) -> None:
        self.level_platform()
        for door_id in self.current_door_state:
            self.close_door(door_id)


if __name__ == "__main__":
    platform = PlatformController()
    platform.tilt_to_slot(1)
    time.sleep(0.2)
    platform.level_platform()

