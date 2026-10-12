"""Streamlit dashboard for the AutoSort system.

The dashboard reads the latest shared camera frame from disk, shows bin fill
levels, and renders a running history of sorted items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image


class AutoSortDashboard:
    def __init__(self, frame_path: str | Path | None = None):
        self.frame_path = (
            Path(frame_path)
            if frame_path
            else Path(__file__).resolve().parent.parent / "tmp" / "autosort_frame.jpg"
        )
        self.fallback_path = Path(__file__).resolve().parent / "fallback.png"
        self.trashcan_id = "🗑️ AutoSort System"
        self.trashcan_location = "Sorting bay / primary intake"
        self.bin_labels = ["Recycling", "Battery", "Compost", "Landfill"]
        self.bin_data = {
            "Recycling": 0.42,
            "Battery": 0.18,
            "Compost": 0.67,
            "Landfill": 0.31,
        }
        self.recent_items: List[Dict[str, Any]] = [
            {"class": "bottle", "confidence": 0.96, "timestamp": "2026-10-11 09:41:02", "bin": "Recycling"},
            {"class": "banana", "confidence": 0.89, "timestamp": "2026-10-11 09:38:45", "bin": "Compost"},
        ]

        if "recent_items" not in st.session_state:
            st.session_state.recent_items = list(self.recent_items)

    @staticmethod
    def _camera_available() -> bool:
        for idx in range(6):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                return True
        return False

    @staticmethod
    def _encoder_available() -> bool:
        try:
            import smbus2
            for bus in (1, 4, 5, 13, 14):
                try:
                    bus_obj = smbus2.SMBus(bus)
                    try:
                        bus_obj.read_byte_data(0x36, 0x0E)
                        return True
                    except Exception:
                        pass
                    finally:
                        bus_obj.close()
                except Exception:
                    continue
        except Exception:
            return False
        return False

    @staticmethod
    def _platform_available() -> bool:
        try:
            import smbus2
            for bus in (1, 4, 5, 13, 14):
                try:
                    bus_obj = smbus2.SMBus(bus)
                    try:
                        bus_obj.read_byte_data(0x40, 0x00)
                        return True
                    except Exception:
                        pass
                    finally:
                        bus_obj.close()
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _load_frame(self):
        if self.frame_path.exists():
            try:
                img = cv2.imread(str(self.frame_path), cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        if self.fallback_path.exists():
            try:
                return np.array(Image.open(self.fallback_path).convert("RGB"))
            except Exception:
                pass

        return None

    def sidebar(self):
        st.sidebar.markdown("# AutoSort Pages")
        page = st.sidebar.radio("Select Page:", ["Dashboard", "Statistics", "Information"])
        st.sidebar.divider()
        return page

    def render_header(self):
        st.title(self.trashcan_id)
        st.markdown(f":violet-badge[Bin location] **{self.trashcan_location}**")

    def render_system_status(self):
        camera_ok = self._camera_available()
        encoder_ok = self._encoder_available()
        platform_ok = self._platform_available()
        status = [
            ("Camera", "live" if camera_ok else "unavailable", "green" if camera_ok else "orange"),
            ("Encoder", "live" if encoder_ok else "missing", "green" if encoder_ok else "orange"),
            ("Servo bus", "ready" if platform_ok else "offline", "green" if platform_ok else "orange"),
        ]

        st.caption("System health")
        cols = st.columns(len(status))
        for col, (label, value, color) in zip(cols, status):
            text = f"{label}: {value}"
            if color == "green":
                col.success(text)
            else:
                col.warning(text)

    def render_camera_feed(self):
        st.subheader("Live Feed")
        frame = self._load_frame()
        if frame is not None:
            st.image(frame, use_container_width=True)
        else:
            st.info("No live frame available. Waiting for the vision pipeline...")

    def render_bin_capacity(self):
        st.subheader("Bin Capacity")
        bin_colors = {
            "Recycling": "🟢",
            "Battery": "🔴",
            "Compost": "🟤",
            "Landfill": "⚫",
        }
        for label, fill in self.bin_data.items():
            fill_ratio = max(0.0, min(float(fill), 1.0))
            icon = bin_colors.get(label, "⬜")
            st.markdown(f"**{icon} {label}**")
            st.progress(fill_ratio)
            st.caption(f"{fill_ratio * 100:.0f}% full")

    def render_recent_items(self):
        st.subheader("Recently Disposed Items")
        items = st.session_state.get("recent_items", [])
        if not items:
            st.caption("No sorted items yet.")
            return

        for item in reversed(items[-10:]):
            conf = item.get("confidence", 0.0)
            label = item.get("class", "unknown")
            timestamp = item.get("timestamp", "unknown")
            bin_name = item.get("bin", "unknown")
            st.markdown(
                f"- **{label}** &nbsp;·&nbsp; {conf:.2f} conf &nbsp;·&nbsp; {bin_name} &nbsp;·&nbsp; `{timestamp}`"
            )

    def render_dev_console(self):
        with st.expander("🛠️ Developer Console"):
            st.write("Latest camera frame:", self.frame_path)
            st.write("Live detection output:", st.session_state.get("last_detection", {}))
            st.write("Encoder position:", st.session_state.get("encoder_position", "unknown"))
            st.write("Platform state:", st.session_state.get("platform_state", "idle"))

    def app(self):
        page = self.sidebar()
        if page == "Dashboard":
            self.render_header()
            self.render_system_status()
            st.markdown("---")
            left_col, right_col = st.columns([1.35, 0.85])
            with left_col:
                self.render_camera_feed()
            with right_col:
                self.render_bin_capacity()
                self.render_recent_items()
            self.render_dev_console()
        elif page == "Statistics":
            st.title("Statistics")
            st.write("Historical fill rates and throughput metrics can be shown here.")
            st.metric("Current throughput", "0 items/min")
            st.metric("Most frequent type", "recycling")
        elif page == "Information":
            st.title("Information")
            st.write("System health, calibration notes, and installation details live here.")
            st.code(
                "AI model: vision_stuff/ai_model\n"
                "Model labels: battery, compost, landfill, recycle\n"
                "Rotation: turntable rotation to target bin\n"
                "Dispense: tilt selected slot, open gate, vibrate, close, reset"
            )


if __name__ == "__main__":
    app = AutoSortDashboard()
    app.app()
