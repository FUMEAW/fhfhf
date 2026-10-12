import time

from hardware.servo_controller import ServoController
from kinematics.stewart_ik import StewartPlatformIK
from hardware.as5600_motor_pid import Motor_control 
import vision_stuff.vision as vision

bin_positions = {"CARDBOARD": 315, "GLASS": 315, "METAL": 315, "PAPER": 315, "BIODEGRADABLE": 135, "PLASTIC": 225}
plat_positions = {1: -45, 2: 45, 3: -135, 4: 135}
class PlatformCommand:
    offset = Motor_control.offset
    def __init__(self, camera_image):
        self.pid = Motor_control()
        self.file = camera_image
        self.servocontrol = ServoController()
        self.plat_ik = StewartPlatformIK(self.servocontrol)

        self.plat_ik.level_platform()

        self.pid.run_pid()
        # TODO: change self.offset after testing. self.offset is hardware offset from what 0 should be
    def operate_platform(self):
        results = vision.results(self.file)
        if not isinstance(results, dict):
            print(f"Vision failed: {results}")
            return

        for slot, material in results.items():
            if slot not in plat_positions or material not in bin_positions:
                print(f"Ignoring invalid detection: slot={slot}, material={material}")
                continue

            self.pid.set_setpoint(
                bin_positions[material] + plat_positions[slot] + self.offset
            )
            self._wait_for_target()
            self.plat_ik.tilt_slot(slot - 1)

            self.pid.set_setpoint(self.offset)
            self._wait_for_target()
            self.plat_ik.level_platform()

    def _wait_for_target(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while not self.pid.target_reached():
            if time.monotonic() >= deadline:
                raise TimeoutError("Motor did not reach its target in time")
            time.sleep(0.01)
