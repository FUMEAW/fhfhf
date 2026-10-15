"""Load and save AutoSort mechanical calibration.

Calibrated values live in hardware/calibration.json and are written by
calibrate_platform.py. Every consumer (PlatformController, the planner) loads
them at startup and falls back to the defaults below for anything missing,
so a missing or partial file never breaks the sorter.

File layout (all keys optional):

    {
      "level_tilt":   [0.0, 0.0, 0.0],             # tilt servos 0-2 when level
      "slot_tilt":    {"1": [-15, 30, 90], ...},   # tilt servos 0-2 per slot
      "flaps":        {"1": {"open": -40, "closed": -75}, ...},  # keyed by SLOT
      "turn_positions": {"0": 0.0, "90": 0.33, "180": 0.67, "270": 1.0},
                                                   # yaw (0-1 on the encoder arc) for each
                                                   # clockwise platform turn; "0" is home
      "camera": {"center": [0.51, 0.625],          # platform hub in the image (0-1)
                 "slots": {"top": 2, "right": 4, "bottom": 3, "left": 1}},
      "pid": {"kp": 0.045, "ki": 0.0, "kd": 0.0}   # yaw PID gains
    }

Why turns instead of bin positions: bins are fixed at the corners, and slots
ride on the platform. To drop slot S into bin B the platform turns by
autosort.config.turn_for(S, B), which is always 0, 90, 180 or 270 degrees.
So four calibrated yaw positions cover every slot/bin pair.

Flaps are keyed by slot number because that is how you see them on the
platform. The slot -> PCA9685 channel mapping comes from hardware.config.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from hardware.config import SLOT_FLAP_CHANNELS

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"

SLOTS = (1, 2, 3, 4)

# Defaults are the values that were hardcoded before calibration existed.
DEFAULT_LEVEL_TILT: List[float] = [0.0, 0.0, 0.0]

DEFAULT_SLOT_TILT: Dict[int, List[float]] = {
    1: [-15.0, 30.0, 90.0],
    2: [-15.0, 90.0, -15.0],
    3: [90.0, 30.0, -15.0],
    4: [90.0, -15.0, 30.0],
}

# Original FLAP_ANGLES were keyed by channel as [open, closed].
_DEFAULT_FLAPS_BY_CHANNEL = {
    4: [8.0, 38.0],
    5: [-48.0, -72.0],
    6: [-44.0, -76.0],
    7: [-40.0, -75.0],
}
DEFAULT_FLAPS: Dict[int, Dict[str, float]] = {
    slot: {
        "open": _DEFAULT_FLAPS_BY_CHANNEL[ch][0],
        "closed": _DEFAULT_FLAPS_BY_CHANNEL[ch][1],
    }
    for slot, ch in SLOT_FLAP_CHANNELS.items()
}

# Placeholder guesses: run calibrate_platform.py and set these for real.
# With the logical 0-270 degree arc these are 0/90/180/270 logical degrees.
DEFAULT_TURN_POSITIONS: Dict[int, float] = {0: 0.0, 90: 0.333, 180: 0.667, 270: 1.0}
TURNS = (0, 90, 180, 270)

# Camera mounted at the landfill corner. The X arms split the image into
# top/right/bottom/left wedges around the platform hub.
DEFAULT_CAMERA = {
    "center": [0.51, 0.625],
    "slots": {"top": 2, "right": 4, "bottom": 3, "left": 1},
}
WEDGES = ("top", "right", "bottom", "left")

# Matches ContinuousRotationServoController's constructor defaults (script.py gains).
DEFAULT_PID = {"kp": 0.045, "ki": 0.0, "kd": 0.0}


def default_calibration() -> Dict[str, Any]:
    """Fresh calibration dict using the built-in defaults."""
    return {
        "level_tilt": list(DEFAULT_LEVEL_TILT),
        "slot_tilt": copy.deepcopy(DEFAULT_SLOT_TILT),
        "flaps": copy.deepcopy(DEFAULT_FLAPS),
        "turn_positions": dict(DEFAULT_TURN_POSITIONS),
        "camera": copy.deepcopy(DEFAULT_CAMERA),
        "pid": dict(DEFAULT_PID),
        "turns_calibrated": False,
    }


def load_calibration(path: Path = CALIBRATION_PATH, verbose: bool = True) -> Dict[str, Any]:
    """Defaults merged with whatever valid values the JSON file provides."""
    cal = default_calibration()
    if not path.exists():
        return cal

    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        if verbose:
            print(f"WARNING: could not read {path.name} ({exc}); using default calibration")
        return cal

    try:
        if "level_tilt" in raw:
            cal["level_tilt"] = _three_floats(raw["level_tilt"])
        for key, angles in raw.get("slot_tilt", {}).items():
            slot = int(key)
            if slot in SLOTS:
                cal["slot_tilt"][slot] = _three_floats(angles)
        for key, flap in raw.get("flaps", {}).items():
            slot = int(key)
            if slot in SLOTS:
                for state in ("open", "closed"):
                    if state in flap:
                        cal["flaps"][slot][state] = float(flap[state])
        turns = raw.get("turn_positions", {})
        for key, pos in turns.items():
            turn = int(key) % 360
            if turn in TURNS:
                cal["turn_positions"][turn] = _unit(pos)
        cal["turns_calibrated"] = all(str(t) in turns for t in TURNS)
        cam = raw.get("camera", {})
        if "center" in cam:
            cx, cy = (float(v) for v in cam["center"])
            cal["camera"]["center"] = [_unit(cx), _unit(cy)]
        if "slots" in cam:
            slots = {w: int(cam["slots"][w]) for w in WEDGES if w in cam["slots"]}
            merged = {**cal["camera"]["slots"], **slots}
            if sorted(merged.values()) != [1, 2, 3, 4]:
                raise ValueError(f"camera slots must use 1-4 once each, got {merged}")
            cal["camera"]["slots"] = merged
        for gain in ("kp", "ki", "kd"):
            if gain in raw.get("pid", {}):
                cal["pid"][gain] = float(raw["pid"][gain])
    except (TypeError, ValueError) as exc:
        if verbose:
            print(f"WARNING: invalid value in {path.name} ({exc}); using defaults for the rest")
    return cal


def save_calibration(cal: Dict[str, Any], path: Path = CALIBRATION_PATH) -> Path:
    """Write calibration atomically so a crash never leaves a half-written file."""
    data = {
        "level_tilt": [round(float(v), 2) for v in cal["level_tilt"]],
        "slot_tilt": {
            str(s): [round(float(v), 2) for v in cal["slot_tilt"][s]] for s in SLOTS
        },
        "flaps": {
            str(s): {
                "open": round(float(cal["flaps"][s]["open"]), 2),
                "closed": round(float(cal["flaps"][s]["closed"]), 2),
            }
            for s in SLOTS
        },
        "turn_positions": {str(t): round(float(cal["turn_positions"][t]), 4) for t in TURNS},
        "camera": {
            "center": [round(float(v), 4) for v in cal["camera"]["center"]],
            "slots": {w: int(cal["camera"]["slots"][w]) for w in WEDGES},
        },
        "pid": {k: round(float(v), 4) for k, v in cal.get("pid", DEFAULT_PID).items()},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def _three_floats(values: Any) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise ValueError(f"expected 3 tilt angles, got {len(vals)}")
    return vals


def _unit(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
