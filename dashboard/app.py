"""Streamlit dashboard for the AutoSort system.

main.py launches this in the background. The dashboard only READS files that
the sorter writes, so it never fights the sorter for the camera or I2C bus:

    tmp/autosort_frame.jpg      annotated camera frame (green = will sort)
    tmp/latest_detection.json   live status: detections, model stats, hardware, yaw
    tmp/sort_history.jsonl      one line per sorted item
    hardware/calibration.json   calibrated angles / bin positions

Live sections refresh every second. Manual yaw control is only enabled while
the sorter is NOT running, because two programs driving the servo at once
would fight each other.

Run alone for testing:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

# Streamlit puts dashboard/ on sys.path, not the repo root; add it so the
# hardware and autosort packages import correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_DIR = ROOT / "tmp"
FRAME_PATH = TMP_DIR / "autosort_frame.jpg"
STATUS_PATH = TMP_DIR / "latest_detection.json"
HISTORY_PATH = TMP_DIR / "sort_history.jsonl"

BINS = ["recycling", "battery", "compost", "landfill"]
BIN_ICONS = {"recycling": "🟢", "battery": "🔴", "compost": "🟤", "landfill": "⚫"}
HEARTBEAT_TIMEOUT_S = 5.0     # status older than this = sorter not updating
SORT_TIMEOUT_S = 60.0         # a single sort sequence can take this long


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def load_status() -> Dict[str, Any]:
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {}


def load_history(limit: int = 2000) -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    items: List[Dict[str, Any]] = []
    try:
        lines = HISTORY_PATH.read_text().splitlines()[-limit:]
    except OSError:
        return []
    for line in lines:
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    return items


def load_frame() -> Optional[np.ndarray]:
    if not FRAME_PATH.exists():
        return None
    img = cv2.imread(str(FRAME_PATH), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sorter_state(status: Dict[str, Any]) -> str:
    """'running', 'sorting', 'dry-run', 'stopped' or 'offline'."""
    if not status:
        return "offline"
    age = time.time() - float(status.get("heartbeat", 0))
    state = status.get("state", "running")
    if state == "stopped":
        return "stopped"
    if state == "sorting" and age < SORT_TIMEOUT_S:
        return "sorting"
    if age < HEARTBEAT_TIMEOUT_S:
        return state
    return "offline"


def real_sorts(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [h for h in history if h.get("ok") and not h.get("dry_run")]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class AutoSortDashboard:
    def __init__(self) -> None:
        self.trashcan_id = "🗑️ AutoSort System"
        self.trashcan_location = "Sorting bay / primary intake"

    def sidebar(self) -> str:
        st.sidebar.markdown("# AutoSort Pages")
        page = st.sidebar.radio("Select Page:", ["Dashboard", "Statistics", "Information"])
        st.sidebar.divider()
        st.sidebar.caption("Live data refreshes every second.")
        return page

    def render_header(self) -> None:
        st.title(self.trashcan_id)
        st.markdown(f":violet-badge[Bin location] **{self.trashcan_location}**")

    # -- live sections (each refreshes itself) -----------------------------

    @st.fragment(run_every=1.0)
    def render_system_status(self) -> None:
        status = load_status()
        state = sorter_state(status)
        hw = status.get("hardware", {}) if state != "offline" else {}
        vision = status.get("vision") or {}

        state_text = {
            "running": "running",
            "sorting": "sorting an item",
            "dry-run": "dry run (no motion)",
            "stopped": "stopped",
            "offline": "not running",
        }[state]
        items = [
            ("Sorter", state_text, state in ("running", "sorting", "dry-run")),
            ("Camera", "live" if hw.get("camera") else "unavailable", bool(hw.get("camera"))),
            ("AI model", "loaded" if vision.get("model_ok") else "not loaded", bool(vision.get("model_ok"))),
            ("Encoder", "live" if hw.get("encoder") else "missing", bool(hw.get("encoder"))),
            ("Servo board", "ready" if hw.get("platform") else "offline", bool(hw.get("platform"))),
        ]
        st.caption("System health")
        cols = st.columns(len(items))
        for col, (label, value, good) in zip(cols, items):
            (col.success if good else col.warning)(f"{label}: {value}")
        if state == "offline":
            st.info("The sorter is not running. Start it with `python main.py`.")

    @st.fragment(run_every=1.0)
    def render_camera_feed(self) -> None:
        st.subheader("Live Feed & AI Detections")
        status = load_status()
        frame = load_frame()
        if frame is not None:
            st.image(frame, width="stretch",
                     caption="Green boxes will be sorted, orange boxes are below the threshold.")
        else:
            st.info("No live frame available. Waiting for the vision pipeline...")

        st.markdown("### 🤖 Live AI Inference")
        det_cols = st.columns(4)
        det_cols[0].metric("Detected Class", status.get("class") or "None")
        det_cols[1].metric("Confidence", f"{float(status.get('confidence') or 0.0):.2f}")
        det_cols[2].metric("Target Bin", status.get("bin") or "None")
        det_cols[3].metric("Slot", status.get("slot") or "-")

        dets = status.get("detections") or []
        threshold = float(status.get("threshold") or 0.0)
        if dets:
            st.dataframe(
                [
                    {
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "slot": d.get("slot"),
                        "bin": d.get("bin"),
                        "will sort": d["confidence"] >= threshold,
                    }
                    for d in dets
                ],
                hide_index=True,
                width="stretch",
            )

        vision = status.get("vision") or {}
        peaks = vision.get("class_peaks") or {}
        if peaks:
            st.caption(
                f"Best score per class on the last frame (threshold {threshold:.2f}). "
                f"Inference {vision.get('last_inference_ms', 0):.0f} ms, "
                f"{vision.get('inference_count', 0)} frames analyzed."
            )
            st.bar_chart(peaks, horizontal=True, height=180)
        if vision.get("error"):
            st.error(f"Vision error: {vision['error']}")

    @st.fragment(run_every=2.0)
    def render_bin_levels(self) -> None:
        st.subheader("Bin Fill Levels")
        bins = load_status().get("bins") or {}
        if not bins:
            st.caption("No fill readings yet. Is the sorter running?")
            return
        for name in BINS:
            r = bins.get(name) or {}
            icon = BIN_ICONS.get(name, "⬜")
            if r.get("fill") is None:
                st.markdown(f"**{icon} {name.title()}**: no reading ({r.get('error') or 'waiting'})")
                st.progress(0.0)
                continue
            tag = " :red-badge[FULL]" if r.get("full") else ""
            st.markdown(f"**{icon} {name.title()}**: {r['fill']:.0%} ({r['distance_mm']:.0f} mm){tag}")
            st.progress(float(r["fill"]))

    @st.fragment(run_every=2.0)
    def render_bin_counts(self) -> None:
        st.subheader("Items Sorted per Bin")
        sorts = real_sorts(load_history())
        counts = Counter(h.get("bin") for h in sorts)
        total = max(1, len(sorts))
        for name in BINS:
            n = counts.get(name, 0)
            st.markdown(f"**{BIN_ICONS.get(name, '⬜')} {name.title()}**: {n} item{'s' if n != 1 else ''}")
            st.progress(n / total)
        st.caption("Counts since the history file was started.")

    @st.fragment(run_every=2.0)
    def render_recent_items(self) -> None:
        st.subheader("Recently Disposed Items")
        history = load_history(limit=50)
        if not history:
            st.caption("No sorted items yet.")
            return
        for item in reversed(history[-10:]):
            tag = " · dry run" if item.get("dry_run") else ("" if item.get("ok") else " · FAILED")
            st.markdown(
                f"- **{item.get('class', 'unknown')}** · {float(item.get('confidence', 0)):.2f} conf · "
                f"{item.get('bin', '?')} via slot {item.get('slot', '?')}{tag} · `{item.get('timestamp', '')}`"
            )

    # -- developer console --------------------------------------------------

    def render_dev_console(self) -> None:
        with st.expander("🛠️ Developer Console & Telemetry"):
            status = load_status()
            state = sorter_state(status)
            st.write("Sorter state:", state)
            st.write("Last sort:", status.get("last_action") or "none yet")
            yaw = status.get("yaw_pos")
            st.write("Yaw position (0-1):", "unknown" if yaw is None else f"{yaw:.3f}")
            st.write("Live status file:", status or "missing")

            st.markdown("---")
            st.subheader("Manual Rotation Test")
            st.caption("To change flap, tilt or bin positions, run `python calibrate_platform.py` on the Pi.")
            if state != "offline" and state != "stopped":
                st.info("Manual rotation is disabled while the sorter is running, so the two don't fight over the servo.")
                return

            from hardware.calibration_store import load_calibration
            cal = load_calibration(verbose=False)
            target = st.slider("Target yaw position (0-1)", 0.0, 1.0, float(cal["turn_positions"][0]), 0.005)
            cols = st.columns(len(cal["turn_positions"]) + 1)
            choice: Optional[float] = None
            if cols[0].button("Move to slider"):
                choice = target
            for col, (turn, pos) in zip(cols[1:], sorted(cal["turn_positions"].items())):
                if col.button(f"turn {turn} ({pos:.2f})"):
                    choice = pos
            if choice is not None:
                with st.spinner(f"Rotating platform to {choice:.3f}..."):
                    try:
                        from hardware.servo_controller import ClosedLoopRotationController
                        ctrl = ClosedLoopRotationController()
                        try:
                            ok = ctrl.move_to_normalized(choice)
                        finally:
                            ctrl.close()
                        if ok:
                            st.success(f"Reached {choice:.3f}")
                        else:
                            st.warning("Motion finished outside tolerance, or the encoder is unavailable.")
                    except Exception as e:
                        st.error(f"Rotation error: {e}")

    # -- other pages --------------------------------------------------------

    def render_statistics(self) -> None:
        st.title("Statistics")
        history = load_history()
        sorts = real_sorts(history)
        if not sorts:
            st.info("No items sorted yet.")
            return
        now = time.time()
        last_hour = [h for h in sorts if now - float(h.get("time", 0)) < 3600]
        classes = Counter(h.get("class") for h in sorts)
        cols = st.columns(4)
        cols[0].metric("Total sorted", len(sorts))
        cols[1].metric("Last hour", len(last_hour))
        cols[2].metric("Most frequent", classes.most_common(1)[0][0])
        durations = [float(h.get("duration_s", 0)) for h in sorts]
        cols[3].metric("Avg sort time", f"{sum(durations) / len(durations):.1f}s")
        failed = [h for h in history if not h.get("ok") and not h.get("dry_run")]
        if failed:
            st.warning(f"{len(failed)} sort sequence(s) failed. See the main.py console for details.")
        st.subheader("Items per bin")
        st.bar_chart(Counter(h.get("bin") for h in sorts))
        st.subheader("Items per slot")
        st.bar_chart({f"slot {k}": v for k, v in sorted(Counter(h.get("slot") for h in sorts).items())})

    def render_information(self) -> None:
        st.title("Information")
        st.code(
            "AI model: vision_stuff/ai_model (RF-DETR small, ONNX)\n"
            "Model labels: battery, compost, landfill, recycle\n"
            "Slot: camera wedge the item sits in (diagonal X arms around the hub)\n"
            "Bin: from the AI label. Bins are fixed at the corners, slots ride on the platform\n"
            "Rotation: turn 0/90/180/270 so the slot is over its bin, PID to the calibrated yaw\n"
            "Dispense: tilt toward slot, open flap, vibrate, close, level, turn home"
        )
        from hardware.calibration_store import CALIBRATION_PATH, load_calibration
        st.subheader("Calibration in use")
        st.caption(
            f"From {CALIBRATION_PATH}" if CALIBRATION_PATH.exists()
            else "No calibration.json yet, using built-in defaults. Run `python calibrate_platform.py`."
        )
        st.json(load_calibration(verbose=False))

    def app(self) -> None:
        page = self.sidebar()
        if page == "Dashboard":
            self.render_header()
            self.render_system_status()
            st.markdown("---")
            left_col, right_col = st.columns([1.35, 0.85])
            with left_col:
                self.render_camera_feed()
            with right_col:
                self.render_bin_levels()
                self.render_bin_counts()
                self.render_recent_items()
            self.render_dev_console()
        elif page == "Statistics":
            self.render_statistics()
        elif page == "Information":
            self.render_information()


if __name__ == "__main__":
    AutoSortDashboard().app()
