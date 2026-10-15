import time
import smbus2
import threading
from adafruit_pca9685 import PCA9685
from adafruit_extended_bus import ExtendedI2C
from simple_pid import PID

from hardware.config import (
    AS5600_ADDRESS,
    AS5600_ANGLE_REGISTER,
    AS5600_I2C_BUS,
    CONTINUOUS_SERVO_CHANNEL,
    PCA9685_ADDRESS,
    SERVO_I2C_BUS,
)


class ServoController:
    """General-purpose PCA9685 servo controller for standard servos."""

    SERVO_FREQ = 50
    SERVO_MIN = 102
    SERVO_MAX = 512
    SERVO_ZERO = 307

    def __init__(self, address=PCA9685_ADDRESS, bus=SERVO_I2C_BUS):
        self.ticks_per_degree = (self.SERVO_MAX - self.SERVO_MIN) / 180.0
        self.address = address
        self.available = False
        self.current_angles = {}
        self.init_error = None

        try:
            self.i2c = ExtendedI2C(bus)
            self.pwm = PCA9685(self.i2c, address=address)
            self.pwm.frequency = self.SERVO_FREQ
            self.available = True
        except Exception as exc:
            self.i2c = None
            self.pwm = None
            self.available = False
            self.init_error = exc

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
        interval_sec: float = 0.020,
    ):
        """Symmetrical vibration routine across multiple servo channels."""
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


class ContinuousRotationServoController:
    """
    Closed-loop PID control for the continuous rotation servo on PCA9685 channel 3.

    Uses the logical-angle scheme from script.py: the AS5600 reading is
    converted to a clockwise "logical" angle starting at RAW_START_DEG, so the
    travel arc runs 0-325 logical degrees with no wrap-around inside it.
    Readings in the dead zone (logical 270-360) are clamped to the nearest end.

    Normalized positions 0-1 map linearly onto logical 0-325 degrees. All
    angles this class returns (read_encoder_angle, spin_to_target results) are
    LOGICAL degrees; use read_raw_angle() for the sensor's own value.
    """

    PCA9685_ADDRESS = PCA9685_ADDRESS
    AS5600_ADDRESS = AS5600_ADDRESS
    AS5600_ANGLE_REGISTER = AS5600_ANGLE_REGISTER
    # Measured on this servo (PCA9685 ticks at 50 Hz): it stays still from
    # about 304 to 328, so the true stop point is ~316, not the nominal 307.
    # set_speed() skips that dead band: any real command starts at
    # NEUTRAL_PULSE +/- DEADBAND_TICKS (about 11 deg/s) and reaches full speed
    # (about 48 deg/s) PULSE_SPAN ticks further out.
    NEUTRAL_PULSE = 316
    DEADBAND_TICKS = 13
    PULSE_SPAN = 22
    MIN_COMMAND = 0.02

    # Zero-crossing arc translation:
    # Physical Start:   0.0 deg (AS5600 ZPOS burned here) -> Logical 0.0 deg
    # Physical End:   325.0 deg -> Logical 325.0 deg
    RAW_START_DEG = 0.0
    LOGICAL_MIN = 0.0
    LOGICAL_MAX = 325.0
    LOGICAL_RANGE = LOGICAL_MAX - LOGICAL_MIN
    # Dead-zone readings above this snap to LOGICAL_MIN, below it to LOGICAL_MAX.
    DEAD_ZONE_SPLIT = 342.5

    # Kept for callers that still use the old names; these are logical degrees.
    ENCODER_MIN = LOGICAL_MIN
    ENCODER_MAX = LOGICAL_MAX
    ENCODER_RANGE = LOGICAL_RANGE

    # Inverted because the 1:2 gear mesh reverses rotation direction
    SERVO_DIRECTION = 1.0

    # Target counts as reached within this many logical degrees, and must stay
    # there for SETTLE_SAMPLES consecutive samples (the servo coasts a little).
    TOLERANCE_DEG = 2.0
    SETTLE_SAMPLES = 3
    # Consecutive failed encoder reads tolerated during a move before aborting.
    MAX_READ_FAILURES = 3

    def __init__(
        self,
        servo_bus=SERVO_I2C_BUS,
        as5600_bus=AS5600_I2C_BUS,
        servo_channel=CONTINUOUS_SERVO_CHANNEL,
        kp=0.045,
        ki=0.0,
        kd=0.0,
    ):
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

        # PID setpoint = 0.0, feeding -error to drive positional error to zero
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=0.0, output_limits=(-1.0, 1.0))
        self.target_normalized = 0.0
        self.current_speed = 0.0

    # -- encoder -------------------------------------------------------------

    def refresh_encoder(self) -> bool:
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

    # The AS5600 on its bit-banged I2C bus often answers 0x0000 for the first
    # one to three reads after a pause. Taken at face value that is a fake
    # position (raw 0 = logical 76 deg) and makes the PID lurch, so zero
    # reads are retried and only trusted when the last good reading was
    # already near raw 0.
    ZERO_READ_RETRIES = 6
    ZERO_NEAR_DEG = 5.0

    def _read_raw_once(self):
        raw = self.encoder_bus.read_i2c_block_data(
            self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2
        )
        return ((raw[0] << 8) | raw[1]) & 0x0FFF

    def try_read_raw_angle(self):
        """Raw AS5600 angle in degrees (0-360), or None if the read failed."""
        if self.encoder_bus is None:
            return None
        last = getattr(self, "_last_raw_deg", None)
        try:
            for _ in range(self.ZERO_READ_RETRIES):
                value = self._read_raw_once()
                if value != 0:
                    break
            else:
                # Every read was zero: believe it if we have no earlier reading
                # (the burned ZPOS makes 0 a real position) or were already near 0 deg.
                near_zero = last is None or min(last, 360.0 - last) <= self.ZERO_NEAR_DEG
                if not near_zero:
                    return None
        except Exception:
            self.encoder_available = False
            return None
        self._last_raw_deg = (value / 4096.0) * 360.0
        return self._last_raw_deg

    def raw_to_logical(self, raw_deg: float) -> float:
        """Clockwise logical angle (0-325) from a raw reading, clamping the dead zone."""
        logical_deg = (raw_deg - self.RAW_START_DEG) % 360.0
        if logical_deg > self.LOGICAL_MAX:
            if logical_deg > self.DEAD_ZONE_SPLIT:  # closer to the 0.0 start
                return self.LOGICAL_MIN
            return self.LOGICAL_MAX               # closer to the 270.0 end
        return logical_deg

    def try_read_encoder_angle(self):
        """Logical angle in degrees, or None if the read failed.

        Use this in control loops: read_encoder_angle() returns LOGICAL_MIN on
        failure, which a PID loop would mistake for a real position.
        """
        raw_deg = self.try_read_raw_angle()
        return None if raw_deg is None else self.raw_to_logical(raw_deg)

    def read_encoder_angle(self) -> float:
        """Logical angle in degrees (LOGICAL_MIN if the read failed)."""
        angle = self.try_read_encoder_angle()
        return self.LOGICAL_MIN if angle is None else angle

    def read_raw_angle(self) -> float:
        """Raw AS5600 angle in degrees (0.0 if the read failed). For diagnostics."""
        angle = self.try_read_raw_angle()
        return 0.0 if angle is None else angle

    # -- conversions -----------------------------------------------------------

    def normalized_to_degrees(self, normalized: float) -> float:
        normalized = max(0.0, min(1.0, float(normalized)))
        return self.LOGICAL_MIN + normalized * self.LOGICAL_RANGE

    def degrees_to_normalized(self, degrees: float) -> float:
        if self.LOGICAL_RANGE <= 0:
            return 0.0
        normalized = (float(degrees) - self.LOGICAL_MIN) / self.LOGICAL_RANGE
        return max(0.0, min(1.0, normalized))

    # -- motion --------------------------------------------------------------------

    def set_speed(self, output: float) -> None:
        """Drive the servo: output -1..1 is mapped past the servo's dead band."""
        speed = max(-1.0, min(1.0, float(output) * self.SERVO_DIRECTION))
        self.current_speed = speed
        if self.pwm is None:
            return
        if abs(speed) < self.MIN_COMMAND:
            pulse = self.NEUTRAL_PULSE
        else:
            sign = 1.0 if speed > 0 else -1.0
            pulse = self.NEUTRAL_PULSE + sign * (self.DEADBAND_TICKS + abs(speed) * self.PULSE_SPAN)
        pulse = max(102, min(512, int(round(pulse))))
        duty_cycle = int((pulse / 4096.0) * 65535)
        self.pwm.channels[self.servo_channel].duty_cycle = duty_cycle

    def set_target_normalized(self, normalized: float) -> None:
        self.target_normalized = max(0.0, min(1.0, float(normalized)))

    def spin_to_target(
        self,
        target_normalized: float = 0.0,
        duration_sec: float = None,
        update_hz: float = 20.0,
    ) -> dict:
        """PID-move to a 0-1 position (logical 0-325 degrees).

        The logical range has no internal wrapping, so the error is a plain
        linear difference. Stops within TOLERANCE_DEG, at the timeout (none if
        duration_sec is None), on Ctrl+C, or if the encoder stops answering.
        """
        start = time.monotonic()
        self.set_target_normalized(target_normalized)
        target_deg = self.normalized_to_degrees(target_normalized)
        samples = []
        read_failures = 0
        encoder_error = False
        settled = 0
        self.pid.reset()

        try:
            while duration_sec is None or time.monotonic() - start < duration_sec:
                current_deg = self.try_read_encoder_angle()
                if current_deg is None:
                    # Never drive on a guessed position: stop, try to reconnect, retry.
                    self.set_speed(0.0)
                    read_failures += 1
                    if read_failures >= self.MAX_READ_FAILURES:
                        encoder_error = True
                        break
                    self.refresh_encoder()
                    time.sleep(1.0 / max(update_hz, 1.0))
                    continue
                read_failures = 0

                error = target_deg - current_deg
                command = self.pid(-error)
                reached = abs(error) <= self.TOLERANCE_DEG
                if reached:
                    command = 0.0

                self.set_speed(command)
                samples.append({
                    "time": round(time.monotonic() - start, 3),
                    "angle_deg": round(current_deg, 2),
                    "normalized_pos": round(self.degrees_to_normalized(current_deg), 3),
                    "error_deg": round(error, 2),
                    "command": round(command, 3),
                })

                # Only finish once the servo has coasted to a stop inside the band.
                settled = settled + 1 if reached else 0
                if settled >= self.SETTLE_SAMPLES:
                    break

                time.sleep(1.0 / max(update_hz, 1.0))

        except KeyboardInterrupt:
            pass
        finally:
            self.set_speed(0.0)

        self.set_speed(0.0)
        final_deg = None if encoder_error else self.try_read_encoder_angle()
        if final_deg is None:
            encoder_error = True
        nan = float("nan")
        return {
            "target_normalized": target_normalized,
            "target_deg": round(target_deg, 2),
            "final_angle_deg": nan if encoder_error else round(final_deg, 2),
            "final_normalized": nan if encoder_error else round(self.degrees_to_normalized(final_deg), 3),
            "encoder_error": encoder_error,
            "samples": samples,
        }

    def stop(self) -> None:
        self._stop_event.set()
        self.set_speed(0.0)

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
    """Yaw platform with continuous-rotation servo on PCA9685 channel 3."""

    def __init__(
        self,
        servo_bus=SERVO_I2C_BUS,
        as5600_bus=AS5600_I2C_BUS,
        servo_channel=CONTINUOUS_SERVO_CHANNEL,
    ):
        self.cont = ContinuousRotationServoController(
            servo_bus=servo_bus,
            as5600_bus=as5600_bus,
            servo_channel=servo_channel,
        )
        self.servo = self.cont
        self.servo_channel = servo_channel
        self.zero_offset_deg = 0.0
        self.motion_min_deg = self.cont.ENCODER_MIN
        self.motion_max_deg = self.cont.ENCODER_MAX
        self.target_deg = self.cont.ENCODER_MIN
        self.tolerance_deg = self.cont.TOLERANCE_DEG
        self.current_angle_deg = self.cont.ENCODER_MIN
        self.encoder_available = self.cont.encoder_available

    def _refresh(self) -> bool:
        self.encoder_available = self.cont.refresh_encoder()
        return self.encoder_available

    def set_zero(self, zero_deg: float = None) -> float:
        self._refresh()
        if zero_deg is None:
            zero_deg = self.read_encoder_angle()
        self.zero_offset_deg = float(zero_deg)
        return self.zero_offset_deg

    def set_motion_range(self, min_deg: float = None, max_deg: float = None) -> None:
        self.motion_min_deg = self.cont.ENCODER_MIN if min_deg is None else float(min_deg)
        self.motion_max_deg = self.cont.ENCODER_MAX if max_deg is None else float(max_deg)

    def read_encoder_angle(self) -> float:
        self._refresh()
        return float(self.cont.read_encoder_angle())

    def heading_to_encoder(self, heading_deg: float) -> float:
        span = self.motion_max_deg - self.motion_min_deg
        t = (float(heading_deg) % 360.0) / 360.0
        return self.motion_min_deg + t * span

    def move_to_normalized(self, normalized: float, timeout: float = 15.0) -> bool:
        self._refresh()
        if not self.encoder_available:
            return False
        result = self.cont.spin_to_target(
            target_normalized=max(0.0, min(1.0, float(normalized))),
            duration_sec=timeout,
        )
        self.target_deg = result["target_deg"]
        if result.get("encoder_error"):
            self.encoder_available = False
            return False
        self.current_angle_deg = result["final_angle_deg"]
        return abs(result["final_angle_deg"] - result["target_deg"]) <= self.tolerance_deg

    def move_to(self, target_deg: float, timeout: float = 8.0, step_delay: float = 0.05) -> bool:
        del step_delay
        encoder_target = self.heading_to_encoder(target_deg)
        normalized = self.cont.degrees_to_normalized(encoder_target)
        return self.move_to_normalized(normalized, timeout=timeout)

    def stop(self) -> None:
        self.cont.stop()

    def close(self):
        self.cont.close()