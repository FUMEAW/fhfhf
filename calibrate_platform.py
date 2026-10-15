#!/usr/bin/env python3
"""Live calibration console for the AutoSort platform.

Run it on the Pi:

    python calibrate_platform.py            # moves real hardware
    python calibrate_platform.py --dry-run  # no hardware, just edit values

Every change moves the hardware immediately so you can see it. Nothing is
written to disk until you type `save`; the sorter (main.py) loads the saved
values from hardware/calibration.json at startup.

What you can calibrate:
  * level      - the 3 tilt servo angles that make the platform flat
  * tilt       - the 3 tilt servo angles for each slot (1-4)
  * flap       - the open and closed angle of each slot's flap
  * turn       - yaw position (0-1 on the encoder arc) for each clockwise
                 platform turn: 0 (home), 90, 180, 270
  * camera     - where the platform hub is in the image and which wedge is which slot
  * pid        - yaw PID gains

How turns work (see "Platform Drawing.png"): bins are fixed at the corners,
slots ride on the platform. At turn 0 (home) slot 1 is over recycling, 2 over
battery, 4 over compost, 3 over landfill. Turn 90 is the position where slot 1
has moved clockwise onto the battery corner. Calibrate each turn by jogging
until the slots line up over the bins: `jog turn 90`.

The easiest way is jog mode, e.g. `jog flap 2 open`, then use the arrow keys.
"""

from __future__ import annotations

import argparse
import copy
import shlex
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from autosort.config import BIN_DIRECTIONS, SLOT_HOME_DIRECTIONS, turn_for
from hardware.calibration_store import (
    CALIBRATION_PATH,
    SLOTS,
    TURNS,
    WEDGES,
    default_calibration,
    load_calibration,
    save_calibration,
)
from hardware.config import SLOT_FLAP_CHANNELS, TILT_CHANNELS

SERVO_LIMIT = 90.0
YAW_DROP_TOLERANCE_DEG = 3.0  # same rule as the sorter: no drop if yaw is further off  # ServoController clamps pulses at roughly +/-90 degrees


class _NullServo:
    """Stand-in servo driver for --dry-run or when the PCA9685 is missing."""

    available = False
    pwm = None

    def __init__(self) -> None:
        self.current_angles: Dict[int, float] = {}

    def set_servo_angle(self, slot: int, angle_deg: float) -> None:
        self.current_angles[slot] = angle_deg

    def vibrate_servos_aggressive(self, slots, amplitude_deg=6.0, duration_sec=1.0, interval_sec=0.02):
        time.sleep(duration_sec)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class PlatformCalibrator:
    """Holds the working calibration and moves the hardware to match it."""

    def __init__(self, dry_run: bool = False, use_yaw: bool = True, servo=None):
        from hardware.platform_control import PlatformController

        self.dry_run = dry_run
        self.cal = load_calibration()
        self._saved = copy.deepcopy(self.cal)

        if servo is not None:
            self.servo = servo  # share an already-open PCA9685 (used by calibration.py)
        elif dry_run:
            self.servo = _NullServo()
        else:
            from hardware.servo_controller import ServoController
            self.servo = ServoController()
            if not self.servo.available:
                print(f"WARNING: PCA9685 not available ({self.servo.init_error}); servo moves are simulated")

        self.rotation = None
        if use_yaw and not dry_run:
            from hardware.servo_controller import ClosedLoopRotationController
            from autosort.planner import apply_pid_tuning
            self.rotation = ClosedLoopRotationController()
            apply_pid_tuning(self.rotation, self.cal["pid"])
            if not self.rotation.encoder_available:
                print("WARNING: AS5600 encoder not available; yaw moves are disabled")

        self.platform = PlatformController(servo_controller=self.servo, calibration=self.cal)

    # -- bookkeeping ------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.cal != self._saved

    def _sync(self) -> None:
        """Push the working values into PlatformController and the PID loop."""
        self.platform.apply_calibration(self.cal)
        if self.rotation is not None:
            from autosort.planner import apply_pid_tuning
            apply_pid_tuning(self.rotation, self.cal["pid"])

    @staticmethod
    def _servo_angle(value: float) -> float:
        return max(-SERVO_LIMIT, min(SERVO_LIMIT, float(value)))

    @staticmethod
    def _check_slot(slot: int) -> int:
        slot = int(slot)
        if slot not in SLOTS:
            hint = ""
            channel_to_slot = {ch: sl for sl, ch in SLOT_FLAP_CHANNELS.items()}
            if slot in channel_to_slot:
                hint = f" (channel {slot} is the flap for slot {channel_to_slot[slot]})"
            raise ValueError(f"slot must be 1-4, got {slot}{hint}")
        return slot

    @staticmethod
    def _check_state(state: str) -> str:
        state = state.lower()
        if state in ("o", "open"):
            return "open"
        if state in ("c", "close", "closed"):
            return "closed"
        raise ValueError("flap state must be 'open' or 'closed'")

    @staticmethod
    def _check_bin(name: str) -> str:
        name = name.lower()
        if name == "recycle":
            name = "recycling"
        if name not in BIN_DIRECTIONS:
            raise ValueError(f"unknown bin '{name}', choose from {sorted(BIN_DIRECTIONS)}")
        return name

    @staticmethod
    def _check_turn(turn) -> int:
        turn = int(float(turn)) % 360
        if turn not in TURNS:
            raise ValueError("turn must be 0, 90, 180 or 270")
        return turn

    # -- moves (use current values) ---------------------------------------

    def move_level(self) -> None:
        for ch, angle in zip(TILT_CHANNELS, self.cal["level_tilt"]):
            self.servo.set_servo_angle(ch, angle)

    def move_tilt(self, slot: int) -> None:
        slot = self._check_slot(slot)
        for ch, angle in zip(TILT_CHANNELS, self.cal["slot_tilt"][slot]):
            self.servo.set_servo_angle(ch, angle)
        # Same as the sorter: only this slot's flap is open, all others closed.
        for other in SLOTS:
            if other != slot:
                self.move_flap(other, "closed")
        self.move_flap(slot, "open")

    def move_flap(self, slot: int, state: str) -> None:
        slot, state = self._check_slot(slot), self._check_state(state)
        self.servo.set_servo_angle(SLOT_FLAP_CHANNELS[slot], self.cal["flaps"][slot][state])

    def close_all_flaps(self) -> None:
        for slot in SLOTS:
            self.move_flap(slot, "closed")

    def move_raw(self, channel: int, angle: float) -> float:
        channel = int(channel)
        if not 0 <= channel <= 15:
            raise ValueError("channel must be 0-15")
        if channel == 3:
            raise ValueError("channel 3 is the continuous yaw servo; use 'yaw' instead")
        angle = self._servo_angle(angle)
        self.servo.set_servo_angle(channel, angle)
        return angle

    def read_yaw(self) -> Optional[Dict[str, float]]:
        if self.rotation is None or not self.rotation._refresh():
            return None
        deg = self.rotation.read_encoder_angle()
        return {"deg": deg, "pos": self.rotation.cont.degrees_to_normalized(deg)}

    def move_yaw(self, pos: float, timeout: float = 6.0) -> bool:
        """PID-move yaw. True if it arrived (or yaw is disabled with --dry-run/--no-yaw)."""
        if self.rotation is None:
            print("  (yaw disabled)")
            return True
        if not self.rotation._refresh():
            print("  (AS5600 encoder unavailable, yaw not moved)")
            return False
        pos = max(0.0, min(1.0, float(pos)))
        ok = self.rotation.move_to_normalized(pos, timeout=timeout)
        if not ok and not self.rotation.encoder_available:
            print("  lost the AS5600 encoder during the move; yaw stopped")
            return False
        here = self.read_yaw() or {}
        miss = abs(float(self.rotation.current_angle_deg) - float(self.rotation.target_deg))
        arrived = ok or miss <= YAW_DROP_TOLERANCE_DEG
        print(
            f"  yaw target {pos:.3f} -> now {here.get('pos', float('nan')):.3f} "
            f"({here.get('deg', float('nan')):.1f} deg){'' if ok else f'  [{miss:.1f} deg off]'}"
        )
        return arrived

    def stop_yaw(self) -> None:
        if self.rotation is not None:
            self.rotation.stop()

    # -- setters (change value, then move so you see it) -------------------

    def set_level(self, index: int, angle: float) -> float:
        index = int(index)
        if index not in (0, 1, 2):
            raise ValueError("tilt servo index must be 0, 1 or 2")
        self.cal["level_tilt"][index] = self._servo_angle(angle)
        self._sync()
        self.move_level()
        return self.cal["level_tilt"][index]

    def set_tilt(self, slot: int, index: int, angle: float) -> float:
        slot, index = self._check_slot(slot), int(index)
        if index not in (0, 1, 2):
            raise ValueError("tilt servo index must be 0, 1 or 2")
        self.cal["slot_tilt"][slot][index] = self._servo_angle(angle)
        self._sync()
        self.move_tilt(slot)
        return self.cal["slot_tilt"][slot][index]

    def set_flap(self, slot: int, state: str, angle: float) -> float:
        slot, state = self._check_slot(slot), self._check_state(state)
        self.cal["flaps"][slot][state] = self._servo_angle(angle)
        self._sync()
        self.move_flap(slot, state)
        return self.cal["flaps"][slot][state]

    def set_turn(self, turn: int, pos: Optional[float] = None, move: bool = True) -> float:
        """Set the yaw position for a platform turn. pos=None stores where yaw is now."""
        turn = self._check_turn(turn)
        if pos is None:
            here = self.read_yaw()
            if here is None:
                raise RuntimeError("encoder unavailable, give a number instead of 'here'")
            pos, move = here["pos"], False
        self.cal["turn_positions"][turn] = max(0.0, min(1.0, float(pos)))
        if move:
            self.move_yaw(self.cal["turn_positions"][turn])
        return self.cal["turn_positions"][turn]

    def move_turn(self, turn: int) -> bool:
        return self.move_yaw(self.cal["turn_positions"][self._check_turn(turn)])

    def align(self, slot: int, bin_name: str) -> int:
        """Turn the platform so `slot` is over `bin_name` (no drop). Returns the turn."""
        slot, name = self._check_slot(slot), self._check_bin(bin_name)
        turn = turn_for(slot, name)
        self.move_turn(turn)
        return turn

    def set_camera_center(self, x: float, y: float) -> List[float]:
        self.cal["camera"]["center"] = [max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))]
        return self.cal["camera"]["center"]

    def set_camera_slots(self, top: int, right: int, bottom: int, left: int) -> Dict[str, int]:
        slots = dict(zip(WEDGES, (int(top), int(right), int(bottom), int(left))))
        if sorted(slots.values()) != [1, 2, 3, 4]:
            raise ValueError("give slots 1-4, each once, in the order top right bottom left")
        self.cal["camera"]["slots"] = slots
        return slots

    def camera_check(self, conf: float = 0.5, out_name: str = "camera_check.jpg") -> List[str]:
        """Grab one camera frame, run the model, and save it with slot wedges drawn.

        Do this with the platform at home (turn 0); that is when the sorter looks.
        """
        import cv2
        from pathlib import Path
        from vision_stuff.vision import YoloDetector, draw_slot_wedges

        det = YoloDetector(conf_threshold=conf)
        det.slot_layout = self.cal["camera"]
        idx = YoloDetector.find_camera_index()
        if idx is None:
            raise RuntimeError("no camera found (is main.py running? it holds the camera)")
        cap = cv2.VideoCapture(idx)
        try:
            frame = None
            for _ in range(5):  # let auto-exposure settle
                ok, f = cap.read()
                if ok:
                    frame = f
        finally:
            cap.release()
        if frame is None:
            raise RuntimeError("camera returned no frame")
        if not det.start_session():
            raise RuntimeError("could not load the ONNX model")
        print("  running the model (can take several seconds on the Pi)...")
        results = det.run_inference(frame)
        draw_slot_wedges(frame, self.cal["camera"])
        lines = []
        for d in results:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, d.center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{d.label} {d.confidence:.2f} -> S{d.slot_num}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            lines.append(f"{d.label} {d.confidence:.2f} at {d.center} -> slot {d.slot_num}")
        out = Path(__file__).resolve().parent / "tmp" / out_name
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), frame)
        lines.append(f"saved {out}")
        return lines

    def set_pid(self, kp: float, ki: float, kd: float) -> Dict[str, float]:
        self.cal["pid"] = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}
        self._sync()
        return self.cal["pid"]

    # -- full-sequence tests ----------------------------------------------

    def test_drop(self, slot: int) -> None:
        """Tilt, open, vibrate, close, level: exactly what the sorter does per slot."""
        slot = self._check_slot(slot)
        self._sync()
        self.platform.expel_item(slot)

    def test_sort(self, bin_name: str, slot: int) -> int:
        """Full sorter sequence for an item in `slot` going to `bin_name`."""
        name, slot = self._check_bin(bin_name), self._check_slot(slot)
        self._sync()
        turn = turn_for(slot, name)
        if not self.move_turn(turn):
            self.move_turn(0)
            raise RuntimeError(f"platform did not reach turn {turn}; drop aborted")
        try:
            self.platform.expel_item(slot)
        finally:
            self.move_turn(0)
        return turn

    def go_home(self) -> None:
        self._sync()
        self.platform.home()
        self.move_turn(0)

    # -- persistence -------------------------------------------------------

    def save(self) -> str:
        path = save_calibration(self.cal)
        self.cal["turns_calibrated"] = True
        self._saved = copy.deepcopy(self.cal)
        return str(path)

    def reload(self) -> None:
        self.cal = load_calibration()
        self._saved = copy.deepcopy(self.cal)
        self._sync()

    def reset_defaults(self) -> None:
        self.cal = default_calibration()
        self._sync()

    def describe(self) -> str:
        c, s = self.cal, self._saved

        def mark(changed: bool) -> str:
            return " *" if changed else ""

        lines = [f"Calibration ({CALIBRATION_PATH.name}){'  [UNSAVED CHANGES]' if self.dirty else ''}"]
        lines.append(f"  level            servos 0/1/2 = {_fmt3(c['level_tilt'])}"
                     f"{mark(c['level_tilt'] != s['level_tilt'])}")
        for slot in SLOTS:
            lines.append(f"  tilt slot {slot}      servos 0/1/2 = {_fmt3(c['slot_tilt'][slot])}"
                         f"{mark(c['slot_tilt'][slot] != s['slot_tilt'][slot])}")
        for slot in SLOTS:
            f = c["flaps"][slot]
            lines.append(f"  flap slot {slot} (ch {SLOT_FLAP_CHANNELS[slot]})  open {f['open']:7.1f}  "
                         f"closed {f['closed']:7.1f}{mark(f != s['flaps'][slot])}")
        for turn in TURNS:
            pos = c["turn_positions"][turn]
            name = "turn 0 (home)" if turn == 0 else f"turn {turn}"
            lines.append(f"  {name:<16} yaw {pos:.3f}{mark(pos != s['turn_positions'][turn])}")
        if not c.get("turns_calibrated") and not self.dirty:
            lines.append("    (turn positions are placeholder guesses until you set and save them)")
        cam = c["camera"]
        lines.append(f"  camera hub       x {cam['center'][0]:.3f}  y {cam['center'][1]:.3f}"
                     f"{mark(cam['center'] != s['camera']['center'])}")
        lines.append("  camera wedges    " + "  ".join(f"{w}=S{cam['slots'][w]}" for w in WEDGES)
                     + mark(cam["slots"] != s["camera"]["slots"]))
        p = c["pid"]
        lines.append(f"  pid              kp {p['kp']}  ki {p['ki']}  kd {p['kd']}{mark(p != s['pid'])}")
        return "\n".join(lines)


def _fmt3(vals: List[float]) -> str:
    return " / ".join(f"{v:6.1f}" for v in vals)


# ---------------------------------------------------------------------------
# Jog mode (arrow keys)
# ---------------------------------------------------------------------------

@dataclass
class JogTarget:
    label: str
    get: Callable[[], float]
    set: Callable[[float], float]   # sets the value AND moves the hardware
    steps: List[float]
    lo: float
    hi: float


def _read_key() -> str:
    """Read one keypress (arrow keys come back as 'up'/'down'/'left'/'right')."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def jog(cal: PlatformCalibrator, target: JogTarget) -> None:
    if not sys.stdin.isatty():
        print("Jog mode needs an interactive terminal.")
        return
    step_i = 1 if len(target.steps) > 1 else 0
    print(
        f"Jogging {target.label}\n"
        "  up/right or + : increase    down/left or - : decrease\n"
        "  [ / ] : smaller / bigger step    s : save    q / Enter : done"
    )

    def show() -> None:
        print(f"\r  {target.label} = {target.get():8.3f}   step {target.steps[step_i]:<6g}      ", end="", flush=True)

    show()
    while True:
        key = _read_key()
        delta = 0.0
        if key in ("up", "right", "+", "="):
            delta = target.steps[step_i]
        elif key in ("down", "left", "-", "_"):
            delta = -target.steps[step_i]
        elif key == "[":
            step_i = max(0, step_i - 1)
        elif key == "]":
            step_i = min(len(target.steps) - 1, step_i + 1)
        elif key == "s":
            print(f"\n  saved to {cal.save()}")
        elif key in ("q", "\r", "\n", "esc", "\x03"):
            print()
            return
        if delta:
            new = max(target.lo, min(target.hi, target.get() + delta))
            if "yaw" in target.label:
                print()  # yaw moves print their own result line
            target.set(new)
        show()


def make_jog_target(cal: PlatformCalibrator, args: List[str]) -> JogTarget:
    servo_steps = [0.5, 1.0, 2.0, 5.0, 10.0]
    yaw_steps = [0.005, 0.01, 0.02, 0.05, 0.1]
    if not args:
        raise ValueError("jog what? e.g. 'jog flap 2 open', 'jog tilt 1 0', 'jog level 2', 'jog bin battery'")
    kind = args[0].lower()
    c = cal.cal
    if kind == "flap" and len(args) == 3:
        slot, state = cal._check_slot(args[1]), cal._check_state(args[2])
        cal.move_flap(slot, state)
        return JogTarget(f"flap {slot} {state}", lambda: c["flaps"][slot][state],
                         lambda v: cal.set_flap(slot, state, v), servo_steps, -SERVO_LIMIT, SERVO_LIMIT)
    if kind == "tilt" and len(args) == 3:
        slot, idx = cal._check_slot(args[1]), int(args[2])
        cal.move_tilt(slot)
        return JogTarget(f"tilt slot {slot} servo {idx}", lambda: c["slot_tilt"][slot][idx],
                         lambda v: cal.set_tilt(slot, idx, v), servo_steps, -SERVO_LIMIT, SERVO_LIMIT)
    if kind == "level" and len(args) == 2:
        idx = int(args[1])
        cal.move_level()
        return JogTarget(f"level servo {idx}", lambda: c["level_tilt"][idx],
                         lambda v: cal.set_level(idx, v), servo_steps, -SERVO_LIMIT, SERVO_LIMIT)
    if kind in ("turn", "home") and len(args) == (2 if kind == "turn" else 1):
        turn = cal._check_turn(args[1]) if kind == "turn" else 0
        cal.move_turn(turn)
        return JogTarget(f"turn {turn} yaw", lambda: c["turn_positions"][turn],
                         lambda v: cal.set_turn(turn, v), yaw_steps, 0.0, 1.0)
    if kind == "pid" and len(args) == 2 and args[1] in ("kp", "ki", "kd"):
        gain = args[1]

        def set_gain(v: float) -> float:
            p = dict(c["pid"])
            p[gain] = round(v, 4)
            cal.set_pid(p["kp"], p["ki"], p["kd"])
            return p[gain]
        return JogTarget(f"pid {gain}", lambda: c["pid"][gain], set_gain,
                         [0.001, 0.005, 0.01, 0.05], 0.0, 2.0)
    raise ValueError("usage: jog flap <slot> open|closed | jog tilt <slot> <0-2> | jog level <0-2> "
                     "| jog turn 0|90|180|270 | jog home | jog pid kp|ki|kd")


# ---------------------------------------------------------------------------
# Command console
# ---------------------------------------------------------------------------

HELP = """
Look / move (uses current values):
  show                               print every value (* = changed, not saved)
  level                              tilt servos to level, all flaps closed
  tilt <slot>                        tilt toward slot 1-4
  flap <slot> open|closed            move one flap
  flaps closed                       close all flaps
  yaw                                read encoder position (0-1 and degrees)
  yaw <0-1>                          PID-move yaw to a position
  turn 0|90|180|270                  PID-move to a calibrated platform turn
  align <slot> <bin>                 turn so that slot is over that bin (no drop)
  table                              which turn every slot/bin pair uses
  home                               level, close flaps, turn 0
  raw <channel> <angle>              move any servo channel, nothing is stored

Define and save (guided, saves automatically when each step is done):
  define flap [slot]                 set open + closed angle of a flap (all 4 if no slot)
  define tilt [slot]                 set the 3 tilt servo angles for a slot (all 4 if no slot)

Change values (moves hardware right away):
  jog flap <slot> open|closed        arrow keys to adjust live  <- easiest
  jog tilt <slot> <servo 0-2>
  jog level <servo 0-2>
  jog turn 0|90|180|270              line the slots up over the bins
  jog pid kp|ki|kd
  set flap <slot> open|closed <angle>
  set tilt <slot> <a0> <a1> <a2>
  set level <a0> <a1> <a2>
  set turn 0|90|180|270 <0-1>|here   'here' stores where yaw is now
  set pid <kp> <ki> <kd>

Camera (platform at turn 0, stop main.py first so the camera is free):
  camera [conf]                      detect on one frame, print slots, save tmp/camera_check.jpg
  set camera center <x 0-1> <y 0-1>  where the X hub is in the image
  set camera slots <top> <right> <bottom> <left>   slot number of each wedge

Test:
  test <slot>                        full drop: tilt, open, vibrate, close, level
  sort <bin> <slot>                  item in <slot> to <bin>: turn, drop, turn home

Files:
  save | reload | reset (defaults, not saved until 'save') | help | quit
""".strip()


class CalibrationConsole:
    def __init__(self, cal: PlatformCalibrator):
        self.cal = cal
        self._quit_warned = False

    def run(self) -> None:
        mode = "DRY RUN (no hardware)" if self.cal.dry_run else "LIVE hardware"
        print(f"AutoSort platform calibration [{mode}]")
        print("Type 'help' for commands. Start with 'show', then e.g. 'jog flap 1 open'.\n")
        print(self.cal.describe())
        while True:
            try:
                raw = input("\ncal (dry run)> " if self.cal.dry_run else "\ncal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                raw = "quit"
            if not raw:
                continue
            try:
                tokens = shlex.split(raw)
            except ValueError as exc:
                print(f"parse error: {exc}")
                continue
            # Allow natural phrasing like "flap slot 2 open" or "tilt slot 1".
            tokens = [tokens[0]] + [t for t in tokens[1:] if t.lower() != "slot"]
            try:
                if self.handle(tokens[0].lower(), tokens[1:]) is False:
                    break
            except ValueError as exc:
                msg = str(exc)
                if "invalid literal" in msg or "could not convert" in msg:
                    msg = "expected a number there"
                print(f"error: {msg}")
                self._print_usage(tokens)
            except KeyboardInterrupt:
                self.cal.stop_yaw()
                print("\ninterrupted")
            except Exception as exc:
                print(f"error: {exc}")
        self.cal.stop_yaw()

    def handle(self, cmd: str, a: List[str]) -> Optional[bool]:
        cal = self.cal
        if cmd != "quit":
            self._quit_warned = False

        if cmd == "help":
            print(HELP)
        elif cmd == "show":
            print(cal.describe())
        elif cmd == "level":
            cal.move_level()
            cal.close_all_flaps()
            print("level, flaps closed")
        elif cmd == "tilt" and len(a) == 1:
            cal.move_tilt(int(a[0]))
            print(f"tilted to slot {a[0]}: {_fmt3(cal.cal['slot_tilt'][int(a[0])])}")
        elif cmd == "flap" and len(a) == 2:
            cal.move_flap(int(a[0]), a[1])
            state = cal._check_state(a[1])
            print(f"flap {a[0]} {state} at {cal.cal['flaps'][int(a[0])][state]:.1f}")
        elif cmd == "flaps" and a == ["closed"]:
            cal.close_all_flaps()
            print("all flaps closed")
        elif cmd == "yaw" and not a:
            here = cal.read_yaw()
            print("encoder unavailable" if here is None else f"yaw {here['pos']:.3f} ({here['deg']:.1f} deg)")
        elif cmd == "yaw" and len(a) == 1:
            cal.move_yaw(float(a[0]))
        elif cmd == "turn" and len(a) == 1:
            cal.move_turn(int(a[0]))
        elif cmd == "align" and len(a) == 2:
            turn = cal.align(int(a[0]), a[1])
            print(f"slot {a[0]} over {a[1]}: turn {turn}")
        elif cmd == "table":
            bins = ["recycling", "battery", "compost", "landfill"]
            print("          " + "".join(f"{b:>11}" for b in bins))
            for slot in sorted(SLOT_HOME_DIRECTIONS):
                print(f"  slot {slot}  " + "".join(f"{turn_for(slot, b):>11}" for b in bins))
        elif cmd == "camera" and len(a) <= 1:
            for line in cal.camera_check(conf=float(a[0]) if a else 0.5):
                print("  " + line)
        elif cmd == "home" and not a:
            cal.go_home()
            print("home")
        elif cmd == "raw" and len(a) == 2:
            print(f"channel {a[0]} -> {cal.move_raw(int(a[0]), float(a[1])):.1f} (not stored)")
        elif cmd == "jog":
            jog(cal, make_jog_target(cal, a))
        elif cmd == "define":
            self._define(a)
        elif cmd == "set":
            self._handle_set(a)
        elif cmd == "test" and len(a) == 1:
            print(f"drop test on slot {a[0]}...")
            cal.test_drop(int(a[0]))
            print("done")
        elif cmd == "sort" and len(a) == 2:
            turn = turn_for(cal._check_slot(int(a[1])), cal._check_bin(a[0]))
            print(f"sort test: slot {a[1]} -> {a[0]} (turn {turn})...")
            cal.test_sort(a[0], int(a[1]))
            print("done")
        elif cmd == "save":
            print(f"saved to {cal.save()}")
        elif cmd == "reload":
            cal.reload()
            print("reloaded from disk")
            print(cal.describe())
        elif cmd == "reset":
            cal.reset_defaults()
            print("reset to built-in defaults (type 'save' to keep)")
            print(cal.describe())
        elif cmd in ("quit", "exit", "q"):
            if cal.dirty and not self._quit_warned:
                self._quit_warned = True
                print("You have unsaved changes. Type 'save', or 'quit' again to discard them.")
                return None
            return False
        else:
            self._print_usage([cmd] + a)
        return None

    @staticmethod
    def _print_usage(tokens: List[str]) -> None:
        """Show the help lines for the command the user tried."""
        words = [t.lower() for t in tokens[:2]]
        lines = [ln for ln in HELP.splitlines() if ln.strip().split()[:1] == words[:1]]
        if words[:1] in (["set"], ["jog"]) and len(words) > 1:
            narrowed = [ln for ln in lines if ln.strip().split()[1:2] == words[1:2]]
            lines = narrowed or lines
        if lines:
            print("usage:")
            for ln in lines:
                print(ln)
        else:
            print("unknown command; type 'help'")

    def _define(self, a: List[str]) -> None:
        """Guided jog through flap open/closed or tilt angles, saving after each slot."""
        cal = self.cal
        if not a or a[0].lower() not in ("flap", "flaps", "tilt"):
            raise ValueError("define what? 'define flap [slot]' or 'define tilt [slot]'")
        if not sys.stdin.isatty():
            print("define needs an interactive terminal (arrow keys). Use 'set flap' / 'set tilt' instead.")
            return
        kind = "tilt" if a[0].lower() == "tilt" else "flap"
        slots = [cal._check_slot(int(a[1]))] if len(a) > 1 else list(SLOTS)
        for slot in slots:
            if kind == "flap":
                for state in ("open", "closed"):
                    print(f"\n== Slot {slot} flap: set the {state.upper()} position ==")
                    jog(cal, make_jog_target(cal, ["flap", str(slot), state]))
                cal.move_flap(slot, "closed")
                f = cal.cal["flaps"][slot]
                print(f"slot {slot} flap: open {f['open']:.1f}, closed {f['closed']:.1f}")
            else:
                for idx in range(3):
                    print(f"\n== Slot {slot} tilt: servo {idx} of 0-2 ==")
                    jog(cal, make_jog_target(cal, ["tilt", str(slot), str(idx)]))
                print(f"slot {slot} tilt: {_fmt3(cal.cal['slot_tilt'][slot])}")
            print(f"saved to {cal.save()}")
            if slot != slots[-1]:
                nxt = input("Enter for next slot, q to stop: ").strip().lower()
                if nxt == "q":
                    break
        cal.move_level()
        cal.close_all_flaps()
        print("done: level, flaps closed")

    def _handle_set(self, a: List[str]) -> None:
        cal = self.cal
        if not a:
            raise ValueError("set what? type 'help'")
        what, rest = a[0].lower(), a[1:]
        if what == "flap" and len(rest) == 3:
            print(f"flap {rest[0]} {rest[1]} = {cal.set_flap(int(rest[0]), rest[1], float(rest[2])):.1f}")
        elif what == "tilt" and len(rest) == 4:
            slot = int(rest[0])
            for i in range(3):
                cal.cal["slot_tilt"][cal._check_slot(slot)][i] = cal._servo_angle(float(rest[1 + i]))
            cal._sync()
            cal.move_tilt(slot)
            print(f"tilt slot {slot} = {_fmt3(cal.cal['slot_tilt'][slot])}")
        elif what == "level" and len(rest) == 3:
            cal.cal["level_tilt"] = [cal._servo_angle(float(v)) for v in rest]
            cal._sync()
            cal.move_level()
            print(f"level = {_fmt3(cal.cal['level_tilt'])}")
        elif what == "turn" and len(rest) == 2:
            pos = None if rest[1].lower() == "here" else float(rest[1])
            print(f"turn {rest[0]} = {cal.set_turn(int(rest[0]), pos):.3f}")
        elif what == "home" and len(rest) == 1:
            pos = None if rest[0].lower() == "here" else float(rest[0])
            print(f"turn 0 (home) = {cal.set_turn(0, pos):.3f}")
        elif what == "camera" and len(rest) == 3 and rest[0] == "center":
            print(f"camera hub = {cal.set_camera_center(float(rest[1]), float(rest[2]))}")
        elif what == "camera" and len(rest) == 5 and rest[0] == "slots":
            print(f"camera wedges = {cal.set_camera_slots(*rest[1:])}")
        elif what == "pid" and len(rest) == 3:
            print(f"pid = {cal.set_pid(*map(float, rest))}")
        else:
            raise ValueError("wrong arguments; type 'help'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AutoSort platform calibration")
    parser.add_argument("--dry-run", action="store_true", help="edit values without touching hardware")
    parser.add_argument("--no-yaw", action="store_true", help="do not use the yaw servo / encoder")
    args = parser.parse_args()
    CalibrationConsole(PlatformCalibrator(dry_run=args.dry_run, use_yaw=not args.no_yaw)).run()


if __name__ == "__main__":
    main()
