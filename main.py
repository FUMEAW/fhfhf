"""Main AutoSort orchestration layer.

Running this file starts both the sorting system and the Streamlit dashboard
in a single command. The dashboard runs as a background subprocess so it does
not block the sort loop.

To run the dashboard alone for testing:
    streamlit run dashboard/app.py

Sort flow (per item, every item on the platform is sorted one by one):
    detect -> slot = camera wedge the item sits in, bin = AI label ->
    turn the platform so that slot is over that bin (0/90/180/270 deg) ->
    tilt toward the slot -> open flap -> vibrate -> close -> level -> turn home

Mechanical values (flap/tilt angles, bin yaw positions, PID gains) are loaded
from hardware/calibration.json. Adjust them live with:
    python calibrate_platform.py

Usage:
    python main.py                      # sort until Ctrl+C, with dashboard
    python main.py --dry-run            # detect and log planned sorts, never move
    python main.py --no-dashboard --conf 0.8

Telemetry:
    A background thread prints every new inference result (label, confidence,
    bbox, slot) plus a periodic status line with inference latency and the
    best per-class score, even when nothing passes the threshold. Every
    inference is also appended as one JSON line to tmp/detections.jsonl.

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from autosort.planner import AutoSortSystem

ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = ROOT / "dashboard" / "app.py"
TELEMETRY_LOG_PATH = ROOT / "tmp" / "detections.jsonl"


class DetectionTelemetry:
    """Background monitor that reports what the vision pipeline is seeing."""

    def __init__(
        self,
        system: AutoSortSystem,
        status_interval: float = 5.0,
        log_path: Optional[Path] = TELEMETRY_LOG_PATH,
    ):
        self.system = system
        self.status_interval = status_interval
        self.log_path = log_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        detector = self.system.detector
        last_count = -1
        last_status = 0.0
        while not self._stop.wait(0.1):
            stats = detector.get_stats()
            now = time.time()

            if stats.inference_count != last_count and stats.inference_count > 0:
                last_count = stats.inference_count
                self._report_inference(stats, detector.get_detections(), detector.conf_threshold)

            if now - last_status >= self.status_interval:
                last_status = now
                self._report_status(stats)

    def _report_inference(self, stats, detections, threshold: float) -> None:
        stamp = time.strftime("%H:%M:%S")
        if detections:
            print(
                f"[{stamp}] #{stats.inference_count} {len(detections)} detection(s) "
                f"in {stats.last_inference_ms:.0f} ms"
            )
            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                bin_name = self.system._resolve_bin(det.label)
                print(
                    f"    {det.label:<10} {det.confidence:5.2f}  "
                    f"bbox=({x1},{y1},{x2},{y2}) center={det.center} "
                    f"slot={det.slot_num} -> bin={bin_name}"
                )
        else:
            peaks = ", ".join(f"{k}={v:.2f}" for k, v in sorted(
                stats.class_peaks.items(), key=lambda kv: -kv[1]))
            print(
                f"[{stamp}] #{stats.inference_count} no detections >= {threshold:.2f} "
                f"in {stats.last_inference_ms:.0f} ms | best scores: {peaks}"
            )

        if self.log_path is not None:
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "inference": stats.inference_count,
                "latency_ms": round(stats.last_inference_ms, 1),
                "threshold": threshold,
                "frame": stats.frame_shape,
                "class_peaks": {k: round(v, 4) for k, v in stats.class_peaks.items()},
                "detections": [
                    {
                        "label": d.label,
                        "class_id": d.class_id,
                        "confidence": round(d.confidence, 4),
                        "bbox": [round(v, 1) for v in d.bbox],
                        "center": list(d.center),
                        "slot": d.slot_num,
                    }
                    for d in detections
                ],
            }
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                print(f"[telemetry] could not write {self.log_path}: {exc}")

    def _report_status(self, stats) -> None:
        age = time.time() - stats.last_inference_time if stats.last_inference_time else None
        age_text = f"{age:.1f}s ago" if age is not None else "never"
        print(
            f"[telemetry] model={'ok' if stats.session_ok else 'FAILED'} "
            f"camera={'ok' if stats.camera_available else 'MISSING'} "
            f"inferences={stats.inference_count} "
            f"avg={stats.avg_inference_ms:.0f} ms last={age_text} "
            f"sorted={self.system.sorted_count}"
            + (f" error={stats.last_error}" if stats.last_error else "")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AutoSort system.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (default: auto-detect)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: planner floor)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Stop after this many fresh detection results, 0 = run until Ctrl+C (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect and print planned sorts without moving any hardware")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Do not launch the Streamlit dashboard")
    parser.add_argument("--status-interval", type=float, default=5.0,
                        help="Seconds between telemetry status lines (default: 5)")
    parser.add_argument("--no-telemetry-log", action="store_true",
                        help="Do not append detections to tmp/detections.jsonl")
    return parser.parse_args()


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
    args = parse_args()
    # Show telemetry immediately even when stdout is piped or redirected
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    dashboard_proc: Optional[subprocess.Popen] = None
    if not args.no_dashboard:
        dashboard_proc = start_dashboard()
        # Give Streamlit a moment to bind its port before the sort loop starts
        time.sleep(2.0)

    system = AutoSortSystem(
        camera_index=args.camera,
        conf_threshold=args.conf,
        dry_run=args.dry_run,
    )

    telemetry = DetectionTelemetry(
        system,
        status_interval=args.status_interval,
        log_path=None if args.no_telemetry_log else TELEMETRY_LOG_PATH,
    )
    try:
        system.start()
        status = system.refresh_hardware_status()
        print(
            f"[telemetry] camera index={status.camera_index} "
            f"threshold={system.detector.conf_threshold:.2f} "
            f"warnings={status.warnings or 'none'}"
        )
        cal_file = ROOT / "hardware" / "calibration.json"
        print(
            f"[telemetry] calibration: {cal_file if cal_file.exists() else 'built-in defaults'} | "
            f"turn positions={system.bridge.turn_positions} "
            f"camera wedges={system.detector.slot_layout['slots']}"
        )
        if args.dry_run:
            print("[telemetry] DRY RUN: hardware will not move")
        if not args.no_telemetry_log:
            print(f"[telemetry] logging detections to {TELEMETRY_LOG_PATH}")
        telemetry.start()
        system.run_loop(max_cycles=args.cycles or None)
    except KeyboardInterrupt:
        print("AutoSort loop interrupted.")
    finally:
        telemetry.stop()
        system.stop()
        if dashboard_proc is not None:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()
            print("Dashboard stopped.")
