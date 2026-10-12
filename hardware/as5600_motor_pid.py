"""Retired DC motor + encoder PID.

Yaw is now the continuous-rotation servo on PCA9685 channel 3
(`hardware.servo_controller.ContinuousRotationServoController`).
This module is kept only for reference and should not be imported by
runtime control paths.
"""

import smbus2
import atexit
import threading
import time
import sys
import os
from simple_pid import PID
from gpiozero import PWMOutputDevice


class Motor_control:
    offset = 0
    # The numeric interval 285..350 is unsafe. The permitted clockwise path
    # is the wraparound arc 350 -> 0 -> 285.
    # Positive motor output is assumed to be clockwise (pin_in1).
    SAFE_START = 350.0
    SAFE_END = 285.0

    def _clockwise_distance(self, start, end):
        return (end - start) % 360.0

    def _safe_arc_length(self):
        return self._clockwise_distance(self.SAFE_START, self.SAFE_END)

    def _safe_position(self, angle):
        """Position along the allowed arc, or None if outside it."""
        position = self._clockwise_distance(self.SAFE_START, angle % 360.0)
        if position <= self._safe_arc_length():
            return position
        return None

    def _clip_to_safe_arc(self, angle):
        """Return the nearest safe endpoint for a target in the unsafe arc."""
        angle %= 360.0
        if self._safe_position(angle) is not None:
            return angle

        return self._nearest_safe_endpoint(angle)

    def _nearest_safe_endpoint(self, angle):
        angle %= 360.0
        to_start = min(
            self._clockwise_distance(angle, self.SAFE_START),
            self._clockwise_distance(self.SAFE_START, angle),
        )
        to_end = min(
            self._clockwise_distance(angle, self.SAFE_END),
            self._clockwise_distance(self.SAFE_END, angle),
        )
        return self.SAFE_START if to_start <= to_end else self.SAFE_END

    def _direct_angle_error(self, target, current):
        return (target - current + 180.0) % 360.0 - 180.0

    def _safe_error(self, _error):
        """Calculate error along the allowed arc, never through the hazard."""
        # Avoid passing None into _safe_position which does angle % 360.0
        if self._current_angle is None or self.setpoint is None:
            return 0.0
        current_position = self._safe_position(self._current_angle)
        target_position = self._safe_position(self.setpoint)
        if current_position is None or target_position is None:
            return 0.0
        return target_position - current_position
    def __init__(self):
        # Define I2C address and bus
        self.AS5600_ADDR = 0x36
        self.ANGLE_REG = 0x0E

        # Change this number to match your software bus ID from 'ls /dev/i2c*'
        SOFTWARE_BUS_ID = 4

        self.setpoint = self._clip_to_safe_arc(self.offset)
        self.pid = PID(2, 0.0, 0.01, self.setpoint, output_limits=(-0.98, 0.98))
        self.pid.error_map = self._safe_error
        self.bus = smbus2.SMBus(SOFTWARE_BUS_ID)

        self.pid.set_auto_mode(True)

        # True if clockwise, false if not but to be tested
        self.last_good_angle = None
        self.DEADBAND = 0.02
        self._current_angle = None

        self.pin_in1 = PWMOutputDevice(17)
        self.pin_in2 = PWMOutputDevice(27)
        self.pin_en = PWMOutputDevice(22)

        self.tolerance = 3.0
        self._stop_event = threading.Event()
        self._pid_thread = None
        self._closed = False
        atexit.register(self.close)

    def read_angle(self):
        # Read two bytes from the angle register
        try:
            raw_data = self.bus.read_i2c_block_data(self.AS5600_ADDR, self.ANGLE_REG, 2)
            angle = (raw_data[0] << 8) | raw_data[1]  # Combine MSB and LSB
            angle = angle & 0x0FFF  # Mask to 12 bits
            angle = (angle / 4096.0) * 360.0
            self.last_good_angle = angle
            return angle # Convert to degrees
        except OSError:
            if self.last_good_angle != None:
                return self.last_good_angle
            raise  
    
    def update_pid(self):
        self._current_angle = self.read_angle()
        control = self.pid(self._current_angle)
        angle_error = self._safe_error(0)

        # The forbidden interval is 285..350. Stop immediately if the sensor
        # is in it, and do not command motion toward either boundary. Using
        # the raw angle here avoids an incorrect wraparound endpoint check.
        angle = self._current_angle
        recovering = False
        if self.SAFE_END <= angle <= self.SAFE_START:
            recovering = True
            recovery_target = self._nearest_safe_endpoint(angle)
            recovery_error = self._direct_angle_error(recovery_target, angle)
            control = max(-0.98, min(0.98, recovery_error * 0.02))

        # Avoid repeatedly applying the motor's minimum PWM for tiny encoder
        # errors, which causes vibration around the target.
        if (not recovering and abs(angle_error) <= self.tolerance) or abs(control) < self.DEADBAND:
            self.pin_in1.value = 0
            self.pin_in2.value = 0
            self.pin_en.value = 1
        elif control > 0:
            self.pin_in1.value = max(control, 180/255)
            self.pin_in2.value = 0
            self.pin_en.value = 1
        else:
            self.pin_in1.value = 0
            self.pin_in2.value = max(abs(control),180/255)
            self.pin_en.value = 1

    def run_pid(self):
        """Start continuously updating the motor controller in the background."""
        if self._pid_thread is not None and self._pid_thread.is_alive():
            return

        self._stop_event.clear()

        def pid_loop():
            try:
                while not self._stop_event.is_set():
                    self.update_pid()
                    self._stop_event.wait(0.02)
            except Exception:
                self.stop_pid()
                # Hardware/I2C errors must not be re-raised from a daemon
                # thread while Python is shutting down.

        self._pid_thread = threading.Thread(target=pid_loop, daemon=True)
        self._pid_thread.start()

    def stop_pid(self):
        """Stop the PID loop and disable motor output."""
        self._stop_event.set()
        self.pin_in1.value = 0
        self.pin_in2.value = 0
        self.pin_en.value = 0
        if (
            self._pid_thread is not None
            and self._pid_thread is not threading.current_thread()
        ):
            self._pid_thread.join(timeout=1.0)

    def close(self):
        """Stop the PID loop and release hardware resources."""
        if self._closed:
            return
        self._closed = True
        self.stop_pid()
        self.pin_in1.close()
        self.pin_in2.close()
        self.pin_en.close()
        self.bus.close()

    def set_setpoint(self, target):
        self.setpoint = self._clip_to_safe_arc(target)
        self.pid.setpoint = self.setpoint

    def target_reached(self):
        return (
            self.pid.error is not None
            and self._current_angle is not None
            and self._safe_position(self._current_angle) is not None
            and abs(self._safe_error(0)) <= self.tolerance
        )


