import time
import smbus2
import threading
from adafruit_pca9685 import PCA9685
from adafruit_extended_bus import ExtendedI2C
from simple_pid import PID


class ServoController:
    """General-purpose PCA9685 servo controller for standard servos."""

    SERVO_FREQ = 50
    SERVO_MIN = 102     # Calibrated safe pulse floor (ticks)
    SERVO_MAX = 512     # Calibrated safe pulse ceiling (ticks)
    SERVO_ZERO = 307    # Default 0-degree reference (ticks, 90° physical)

    def __init__(self, address=0x40, bus=1):
        self.ticks_per_degree = (self.SERVO_MAX - self.SERVO_MIN) / 180.0
        self.address = address
        self.available = False
        self.current_angles = {}

        try:
            self.i2c = ExtendedI2C(bus)
            self.pwm = PCA9685(self.i2c, address=address)
            self.pwm.frequency = self.SERVO_FREQ
            self.available = True
        except Exception:
            self.i2c = None
            self.pwm = None
            self.available = False

    def set_servo_angle(self, slot: int, angle_deg: float):
        """Set target angle for a specific PCA9685 channel."""
        self.current_angles[slot] = angle_deg
        if self.pwm is None:
            return
        pulse = int(self.SERVO_ZERO - (angle_deg * self.ticks_per_degree))
        pulse = max(self.SERVO_MIN, min(self.SERVO_MAX, pulse))
        duty_cycle = int((pulse / 4096.0) * 65535)
        self.pwm.channels[slot].duty_cycle = duty_cycle

    def vibrate_servos_aggressive(
        self,
        slots: list,
        amplitude_deg: float = 6.0,
        duration_sec: float = 1.0,
        interval_sec: float = 0.020
    ):
        base_positions = {s: self.current_angles.get(s, 0.0) for s in slots}
        start_time = time.time()
        toggle = True

        while (time.time() - start_time) < duration_sec:
            for idx, s in enumerate(slots):
                phase_mult = 1.0 if (idx % 2 == 0) else -1.0
                sign = phase_mult if toggle else -phase_mult
                target = base_positions[s] + (sign * amplitude_deg)
                pulse = int(self.SERVO_ZERO - (target * self.ticks_per_degree))
                pulse = max(self.SERVO_MIN, min(self.SERVO_MAX, pulse))
                if self.pwm:
                    self.pwm.channels[s].duty_cycle = int((pulse / 4096.0) * 65535)
            toggle = not toggle
            time.sleep(interval_sec)

        for s in slots:
            self.set_servo_angle(s, base_positions[s])


class Servo3Controller:
    """Servo on PCA9685 channel 3 with AS5600 magnetic encoder feedback."""

    SERVO_CHANNEL = 3
    PCA9685_ADDRESS = 0x40
    AS5600_ADDRESS = 0x36
    AS5600_ANGLE_REGISTER = 0x0E

    def __init__(self, servo_bus=1, as5600_bus=4):
        self.servo_bus = servo_bus
        self.as5600_bus_id = as5600_bus
        self.encoder_available = False
        self.servo_available = False

        try:
            self.servo_i2c = ExtendedI2C(self.servo_bus)
            self.pwm = PCA9685(self.servo_i2c, address=self.PCA9685_ADDRESS)
            self.pwm.frequency = 50
            self.servo_available = True
        except Exception:
            self.servo_i2c = None
            self.pwm = None
            self.servo_available = False

        try:
            self.as5600_bus = smbus2.SMBus(self.as5600_bus_id)
            self.as5600_bus.read_i2c_block_data(self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2)
            self.encoder_available = True
        except Exception:
            self.as5600_bus = None
            self.encoder_available = False

        self.servo_min = 102
        self.servo_max = 512
        self.servo_zero = 307
        self.ticks_per_degree = (self.servo_max - self.servo_min) / 180.0

        self.target_angle = 114.0
        self.current_angle = 114.0
        self.last_good_angle = None
        self.tolerance = 2.0

    def read_as5600_angle(self) -> float:
        """Return AS5600 encoder angle in degrees (0-359.9)."""
        if not self.encoder_available or self.as5600_bus is None:
            return self.last_good_angle if self.last_good_angle is not None else 114.0
        try:
            raw = self.as5600_bus.read_i2c_block_data(
                self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2
            )
            raw_value = ((raw[0] << 8) | raw[1]) & 0x0FFF
            angle = (raw_value / 4096.0) * 360.0
            self.last_good_angle = angle
            return angle
        except OSError:
            self.encoder_available = False
            return self.last_good_angle if self.last_good_angle is not None else 114.0

    def set_servo_pwm(self, angle_deg: float):
        """Write a servo angle to PCA9685 channel 3."""
        if self.pwm is None:
            return
        target = max(114.0, min(229.0, float(angle_deg)))
        pulse = int(self.servo_zero - (target * self.ticks_per_degree))
        pulse = max(self.servo_min, min(self.servo_max, pulse))
        duty = int((pulse / 4096.0) * 65535)
        self.pwm.channels[self.SERVO_CHANNEL].duty_cycle = duty

    def move_to_angle(self, target_deg: float, timeout: float = 5.0, step_delay: float = 0.03) -> bool:
        """Rotate servo 3 toward a target angle using AS5600 feedback."""
        self.target_angle = max(114.0, min(229.0, float(target_deg)))
        start = time.time()
        while (time.time() - start) < timeout:
            self.current_angle = self.read_as5600_angle()
            error = (self.target_angle - self.current_angle + 180.0) % 360.0 - 180.0
            if abs(error) <= self.tolerance:
                self.set_servo_pwm(self.target_angle)
                return True
            command_angle = self.current_angle + max(-25.0, min(25.0, error * 1.2))
            self.set_servo_pwm(command_angle)
            time.sleep(step_delay)
        return False

    def hold_angle(self, angle_deg: float):
        """Hold servo at the given angle."""
        self.target_angle = max(114.0, min(229.0, float(angle_deg)))
        self.set_servo_pwm(self.target_angle)

    def close(self):
        """Release PCA9685 and AS5600 resources."""
        try:
            self.pwm.deinit()
        except Exception:
            pass
        try:
            self.as5600_bus.close()
        except Exception:
            pass


class ContinuousRotationServoController:
    """
    Closed-loop PID control for a continuous rotation servo on PCA9685 channel 3
    using the standard Python `simple_pid` library combined with the custom loopback error function.
    """

    PCA9685_ADDRESS = 0x40
    AS5600_ADDRESS = 0x36
    AS5600_ANGLE_REGISTER = 0x0E
    NEUTRAL_PULSE = 307
    PULSE_SPAN = 110

    ENCODER_MIN = 114.0
    ENCODER_MAX = 229.0
    ENCODER_RANGE = ENCODER_MAX - ENCODER_MIN

    FORBIDDEN_CCW_START = 230.0
    FORBIDDEN_CCW_END = 90.0

    def __init__(self, servo_bus=1, as5600_bus=4, servo_channel=3, kp=0.8, ki=0.0, kd=0.3):
        self.servo_bus = servo_bus
        self.as5600_bus_id = as5600_bus
        self.servo_channel = servo_channel
        self.servo_available = False
        self.encoder_available = False
        self.pwm = None
        self.encoder_bus = None
        self._stop_event = threading.Event()
        self._loop_thread = None

        try:
            self.pwm_i2c = ExtendedI2C(self.servo_bus)
            self.pwm = PCA9685(self.pwm_i2c, address=self.PCA9685_ADDRESS)
            self.pwm.frequency = 50
            self.servo_available = True
        except Exception:
            self.pwm = None
            self.servo_available = False

        try:
            self.encoder_bus = smbus2.SMBus(self.as5600_bus_id)
            self.encoder_bus.read_i2c_block_data(self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2)
            self.encoder_available = True
        except Exception:
            self.encoder_bus = None
            self.encoder_available = False

        # Setpoint = 0.0. We feed -error into self.pid() to prevent control inversion.
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=0.0, output_limits=(-1.0, 1.0))
        self.target_normalized = 0.0
        self.current_speed = 0.0

    def _safe_error(self, current_deg: float, target_deg: float) -> float:
        """
        Loopback / safety function: compute signed error that routes correctly 
        through boundaries and avoids forbidden regions.
        """
        current = current_deg % 360.0
        target = target_deg % 360.0

        error = target - current
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360

        if error < 0:
            ccw_destination = (current + error) % 360.0
            if current >= self.FORBIDDEN_CCW_START and ccw_destination <= self.FORBIDDEN_CCW_END:
                error = 360.0 + error
            elif current >= self.FORBIDDEN_CCW_START and ccw_destination < current:
                error = 360.0 + error

        return error

    def refresh_encoder(self) -> bool:
        """Re-open encoder connection if lost."""
        if self.encoder_available and self.encoder_bus is not None:
            return True
        try:
            if self.encoder_bus is not None:
                try:
                    self.encoder_bus.close()
                except Exception:
                    pass
            self.encoder_bus = smbus2.SMBus(self.as5600_bus_id)
            self.encoder_bus.read_i2c_block_data(self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2)
            self.encoder_available = True
            return True
        except Exception:
            self.encoder_bus = None
            self.encoder_available = False
            return False

    def normalized_to_degrees(self, normalized: float) -> float:
        """Convert 0-1 normalized value to encoder degrees."""
        normalized = max(0.0, min(1.0, float(normalized)))
        return self.ENCODER_MIN + normalized * self.ENCODER_RANGE

    def degrees_to_normalized(self, degrees: float) -> float:
        """Convert encoder degrees to 0-1 normalized value."""
        if self.ENCODER_RANGE <= 0:
            return 0.0
        normalized = (float(degrees) - self.ENCODER_MIN) / self.ENCODER_RANGE
        return max(0.0, min(1.0, normalized))

    def set_pid(self, kp: float, ki: float, kd: float) -> None:
        self.pid.tunings = (kp, ki, kd)

    def read_encoder_angle(self) -> float:
        if self.encoder_bus is None:
            return self.ENCODER_MIN
        try:
            raw = self.encoder_bus.read_i2c_block_data(
                self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2
            )
            raw_value = ((raw[0] << 8) | raw[1]) & 0x0FFF
            return (raw_value / 4096.0) * 360.0
        except Exception:
            self.encoder_available = False
            return self.ENCODER_MIN

    def set_speed(self, output: float) -> None:
        if self.pwm is None:
            return
        speed = max(-1.0, min(1.0, float(output)))
        pulse = self.NEUTRAL_PULSE + int(speed * self.PULSE_SPAN)
        pulse = max(102, min(512, pulse))
        duty_cycle = int((pulse / 4096.0) * 65535)
        self.pwm.channels[self.servo_channel].duty_cycle = duty_cycle
        self.current_speed = speed

    def set_target_normalized(self, normalized: float) -> None:
        """Set target using 0-1 normalized input."""
        self.target_normalized = max(0.0, min(1.0, float(normalized)))

    def spin_to_target(
        self,
        target_normalized: float = 0.0,
        duration_sec: float = 3.0,
        update_hz: float = 20.0
    ) -> dict:
        """
        Spin servo to a normalized target position using closed-loop PID and loopback.
        Returns a dict with final position and per-step samples.
        """
        start = time.monotonic()
        self.set_target_normalized(target_normalized)
        target_deg = self.normalized_to_degrees(target_normalized)
        samples = []

        while time.monotonic() - start < duration_sec:
            current_deg = self.read_encoder_angle()
            current_norm = self.degrees_to_normalized(current_deg)
            error = self._safe_error(current_deg, target_deg)

            # Pass -error into PID library to match setpoint=0 calculation correctly
            command = self.pid(-error)

            if abs(error) <= 0.5:
                command = 0.0

            self.set_speed(command)

            samples.append({
                "time": round(time.monotonic() - start, 3),
                "angle_deg": round(current_deg, 2),
                "normalized_pos": round(current_norm, 3),
                "error_deg": round(error, 2),
                "command": round(command, 3)
            })

            if abs(error) <= 0.5:
                break

            time.sleep(1.0 / max(update_hz, 1.0))

        self.set_speed(0.0)
        final_deg = self.read_encoder_angle()
        return {
            "target_normalized": target_normalized,
            "target_deg": round(target_deg, 2),
            "final_angle_deg": round(final_deg, 2),
            "final_normalized": round(self.degrees_to_normalized(final_deg), 3),
            "samples": samples,
        }

    def start_pid_loop(self, target_normalized: float = 0.0) -> None:
        """Start a background PID loop holding a normalized target position."""
        self.set_target_normalized(target_normalized)
        self._stop_event.clear()
        target_deg = self.normalized_to_degrees(self.target_normalized)

        def _loop():
            nonlocal target_deg
            while not self._stop_event.is_set():
                target_deg = self.normalized_to_degrees(self.target_normalized)
                current_deg = self.read_encoder_angle()
                error = self._safe_error(current_deg, target_deg)
                command = self.pid(-error)

                if abs(error) <= 0.5:
                    command = 0.0

                self.set_speed(command)
                time.sleep(0.02)
            self.set_speed(0.0)

        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.set_speed(0.0)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.pwm.deinit()
        except Exception:
            pass
        try:
            self.encoder_bus.close()
        except Exception:
            pass


class ClosedLoopRotationController:
    """
    High-level closed-loop rotation controller using AS5600 feedback.
    Supports runtime zero calibration and configurable motion range.
    """

    def __init__(self, servo_bus=1, as5600_bus=4, servo_channel=3):
        self.servo = Servo3Controller(servo_bus=servo_bus, as5600_bus=as5600_bus)
        self.servo_channel = servo_channel
        self.zero_offset_deg = 0.0
        self.motion_min_deg = 114.0
        self.motion_max_deg = 229.0
        self.target_deg = 114.0
        self.tolerance_deg = 1.0
        self.max_step_deg = 25.0
        self.current_angle_deg = 114.0
        self.encoder_available = self.servo.encoder_available

    @staticmethod
    def _wrap_degrees(value_deg: float) -> float:
        return ((value_deg + 180.0) % 360.0) - 180.0

    def set_zero(self, zero_deg: float = None) -> float:
        """
        Set encoder zero point. Pass a value to set manually,
        or leave None to use the current encoder reading as zero.
        """
        if zero_deg is None:
            zero_deg = self.read_encoder_angle()
        self.zero_offset_deg = float(zero_deg)
        self.encoder_available = self.servo.encoder_available
        return self.zero_offset_deg

    def set_motion_range(self, min_deg: float = 114.0, max_deg: float = 229.0) -> None:
        """Set the usable motion range in degrees."""
        self.motion_min_deg = float(min_deg)
        self.motion_max_deg = float(max_deg)
        self.target_deg = self._clamp_target(self.target_deg)

    def read_encoder_angle(self) -> float:
        return float(self.servo.read_as5600_angle())

    def _clamp_target(self, target_deg: float) -> float:
        target_deg = float(target_deg)
        low = min(self.motion_min_deg, self.motion_max_deg)
        high = max(self.motion_min_deg, self.motion_max_deg)
        return min(max(target_deg, low), high)

    def _error_to_target(self, target_deg: float, current_deg: float) -> float:
        return self._wrap_degrees(target_deg - current_deg)

    def move_to(self, target_deg: float, timeout: float = 8.0, step_delay: float = 0.05) -> bool:
        """Move to a target angle in degrees, respecting zero offset and motion range."""
        self.encoder_available = self.servo.encoder_available
        self.target_deg = self._clamp_target(float(target_deg))

        if not self.encoder_available:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.read_encoder_angle() - self.zero_offset_deg
            error = self._error_to_target(self.target_deg, current)

            if abs(error) <= self.tolerance_deg:
                self.servo.set_servo_pwm(self.target_deg)
                self.current_angle_deg = current
                return True

            command = current + max(-self.max_step_deg, min(self.max_step_deg, error * 0.8))
            self.servo.set_servo_pwm(command)
            self.current_angle_deg = current
            time.sleep(step_delay)

        self.servo.set_servo_pwm(self.target_deg)
        return False

    def stop(self) -> None:
        self.servo.set_servo_pwm(self.current_angle_deg)

    def close(self):
        self.servo.close()


# Backward-compatible aliases
Servo3AS5600Controller = Servo3Controller
ClosedLoopServoController = ClosedLoopRotationController