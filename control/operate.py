import time

from autosort.config import BIN_ANGLES, LABEL_ALIASES, SLOT_MAP, SORTING_RULES
from hardware.platform_control import PlatformController
from hardware.servo_controller import ClosedLoopRotationController
from vision_stuff.vision import YoloDetector


class PlatformCommand:
    def __init__(self, camera_image=None):
        self.file = camera_image
        self.rotation = ClosedLoopRotationController()
        self.platform = PlatformController()
        self.detector = YoloDetector(camera_index=None)

        # Degrade gracefully if encoder is missing rather than crashing at startup
        if not self.rotation.encoder_available:
            print("WARNING: AS5600 encoder not found on I2C bus. Rotation will be open-loop.")
        
        self.platform.level_platform()
        
        if self.rotation.encoder_available:
            self.rotation.move_to(0.0)

    def operate_platform(self):
        self.detector.start()
        try:
            deadline = time.monotonic() + 2.0
            detections = []
            while time.monotonic() < deadline:
                detections = self.detector.get_detections()
                if detections:
                    break
                time.sleep(0.05)
        finally:
            self.detector.stop()

        if not detections:
            print("Vision failed: no detections")
            return

        for det in detections:
            material = (det.label or "").lower().strip()
            mapped = LABEL_ALIASES.get(material, material)
            if mapped is None:
                continue
            bin_name = SORTING_RULES.get(mapped, mapped)
            slot = det.slot_num or SLOT_MAP.get(bin_name)
            if slot is None or bin_name not in BIN_ANGLES:
                print(f"Ignoring invalid detection: slot={slot}, material={material}")
                continue

            if self.rotation.encoder_available:
                self.rotation.move_to(BIN_ANGLES[bin_name])
            self.platform.expel_item(slot)
            if self.rotation.encoder_available:
                self.rotation.move_to(0.0)
            self.platform.level_platform()
