#!/usr/bin/env python3
"""Simple encoder position reader - no heavy dependencies."""

import sys
import time
from hardware.servo_controller import Servo3Controller, ContinuousRotationServoController

def main():
    servo = Servo3Controller(servo_bus=1, as5600_bus=4)
    cont = ContinuousRotationServoController(servo_bus=1, as5600_bus=4, servo_channel=3)
    
    if not servo.encoder_available:
        print("AS5600 encoder not found on bus 4")
        sys.exit(1)
    
    # Single read
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        print(f"Watching encoder (interval: {interval}s). Press Ctrl+C to stop.")
        print(f"Range: {cont.ENCODER_MIN:.2f}° (0.0) to {cont.ENCODER_MAX:.2f}° (1.0)")
        try:
            while True:
                angle_deg = servo.read_as5600_angle()
                normalized = cont.degrees_to_normalized(angle_deg)
                print(f"Position: {angle_deg:7.2f}° | Normalized: {normalized:.3f} (0-1)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        angle_deg = servo.read_as5600_angle()
        normalized = cont.degrees_to_normalized(angle_deg)
        print(f"Range: {cont.ENCODER_MIN:.2f}° (0.0) to {cont.ENCODER_MAX:.2f}° (1.0)")
        print(f"Position: {angle_deg:.2f}° | Normalized: {normalized:.3f} (0-1)")

if __name__ == "__main__":
    main()
