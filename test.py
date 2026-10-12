import time
from hardware.servo_controller import ClosedLoopRotationController

controller = ClosedLoopRotationController()

try:
    print("AS5600 bus:", controller.cont.as5600_bus_id)
    print("Starting encoder angle:", controller.read_encoder_angle())
    controller.move_to(90, timeout=5.0)
    time.sleep(1.0)
    controller.move_to(180, timeout=5.0)
    time.sleep(1.0)
    controller.move_to(0, timeout=5.0)
    print("Final encoder angle:", controller.read_encoder_angle())
finally:
    controller.close()
