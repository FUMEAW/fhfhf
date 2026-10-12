import time
from hardware.servo_controller import Servo3Controller

servo = Servo3Controller(servo_bus=1, as5600_bus=4)

try:
    print("Servo 3 starting at AS5600 angle:", servo.read_as5600_angle())
    servo.move_to_angle(90, timeout=5.0)
    time.sleep(1.0)
    servo.move_to_angle(180, timeout=5.0)
    time.sleep(1.0)
    servo.move_to_angle(0, timeout=5.0)
    print("Final AS5600 angle:", servo.read_as5600_angle())
finally:
    servo.close()
    
