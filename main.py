"""Main AutoSort orchestration layer.

Running this file starts both the sorting system and the Streamlit dashboard
in a single command. The dashboard runs as a background subprocess so it does
not block the sort loop.

To run the dashboard alone for testing:
    streamlit run dashboard/app.py

Sort flow:
    classify -> rotate to bin -> tilt to slot -> open door -> eject -> close door -> level -> home
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from autosort.planner import AutoSortSystem

DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard" / "app.py"


def start_dashboard() -> subprocess.Popen:
    """Launch the Streamlit dashboard as a non-blocking background process."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH),
         "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Dashboard started (PID {proc.pid}) — http://localhost:8501")
    return proc


if __name__ == "__main__":
    dashboard_proc = start_dashboard()
    # Give Streamlit a moment to bind its port before the sort loop starts
    time.sleep(2.0)

    system = AutoSortSystem(camera_index=None)
    try:
        system.start()
        system.run_loop(max_cycles=200)
    except KeyboardInterrupt:
        print("AutoSort loop interrupted.")
    finally:
        system.stop()
        dashboard_proc.terminate()
        try:
            dashboard_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            dashboard_proc.kill()
        print("Dashboard stopped.")
