"""Platform motion helpers for the AutoSort mechanism.

This module keeps the Stewart-platform orientation, door actuation, and platform
leveling in one place with explicit, well-named operations.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from hardware.servo_controller import ServoController


class PlatformController:
    """Drive tilt, door, and stable level logic for the sorting platform."""

    SLOT_ANGLES = {
        1: [-15.0, 30.0, 90.0],
        2: [-15.0, 90.0, -15.0],
        3: [90.0, 30.0, -15.0],
        4: [90.0, -15.0, 30.0],
    }

    SLOT_FLAP_MAP = {1: 7, 2: 5, 3: 4, 4: 6}

    FLAP_ANGLES = {
        4: [8.0, 38.0],
        5: [-48.0, -72.0],
        6: [-44.0, -76.0],
        7: [-40.0, -75.0],
    }

    DOOR_ANGLES = {
        1: [20.0, 90.0],
        2: [20.0, 90.0],
        3: [20.0, 90.0],
        4: [20.0, 90.0],
    }

    def __init__(self, servo_controller: Optional[ServoController] = None):
        self.controller = servo_controller or ServoController()
        self.current_slot = None
        self.current_door_state = {1: "closed", 2: "closed", 3: "closed", 4: "closed"}

    def tilt_to_slot(self, slot: int) -> None:
        if slot not in self.SLOT_ANGLES:
            raise ValueError(f"Invalid slot: {slot}. Must be between 1 and 4.")

        angles = self.SLOT_ANGLES[slot]
        for servo_index, angle in enumerate(angles):
            self.controller.set_servo_angle(servo_index, angle)

        flap_num = self.SLOT_FLAP_MAP[slot]
        self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][0])
        self.current_slot = slot

    def level_platform(self) -> None:
        for servo_index in range(3):
            self.controller.set_servo_angle(servo_index, 0.0)

        for flap_num in self.SLOT_FLAP_MAP.values():
            self.controller.set_servo_angle(flap_num, self.FLAP_ANGLES[flap_num][1])
        self.current_slot = None

    def open_door(self, door_id: int) -> None:
        if door_id not in self.DOOR_ANGLES:
            raise ValueError(f"Invalid door_id: {door_id}. Must be between 1 and 4.")
        self.controller.set_servo_angle(door_id, self.DOOR_ANGLES[door_id][0])
        self.current_door_state[door_id] = "open"

    def close_door(self, door_id: int) -> None:
        if door_id not in self.DOOR_ANGLES:
            raise ValueError(f"Invalid door_id: {door_id}. Must be between 1 and 4.")
        self.controller.set_servo_angle(door_id, self.DOOR_ANGLES[door_id][1])
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

