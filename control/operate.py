"""One-shot sort: detect what is on the platform once, sort it, and stop.

Uses the same planning and motion code as main.py (AutoSortSystem), so slot
detection, slot-to-bin turns and calibration always match the main sorter.
"""

import time

from autosort.planner import AutoSortSystem


class PlatformCommand:
    def __init__(self, camera_image=None, camera_index=None, dry_run=False):
        self.file = camera_image
        self.system = AutoSortSystem(camera_index=camera_index, dry_run=dry_run)
        # Kept for backwards compatibility with older callers
        self.rotation = self.system.rotation
        self.platform = self.system.platform
        self.detector = self.system.detector

        if not self.rotation.encoder_available:
            print("WARNING: AS5600 encoder not found on I2C bus. Rotation is disabled.")

    def operate_platform(self, timeout: float = 30.0):
        """Wait for one fresh detection result, then sort every item found."""
        self.system.start()
        try:
            started = time.time()
            detections = self.detector.wait_for_detections_since(started, timeout=timeout)
            if detections is None:
                print("Vision failed: no fresh detection result")
                return []
            actions = self.system.plan_actions(detections)
            if not actions:
                print("Nothing above the confidence threshold to sort")
                return []
            for action in actions:
                self.system.execute_action(action)
            return actions
        finally:
            self.system.stop()


if __name__ == "__main__":
    PlatformCommand().operate_platform()
