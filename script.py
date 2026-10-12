import time
import smbus2
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

class ContinuousRotationServoControllerClean:
    PCA9685_ADDRESS = PCA9685_ADDRESS
    AS5600_ADDRESS = AS5600_ADDRESS
    AS5600_ANGLE_REGISTER = AS5600_ANGLE_REGISTER
    NEUTRAL_PULSE = 307
    PULSE_SPAN = 110

    # Zero-crossing arc translation:
    # Physical Start: 220.0° -> Logical 0.0°
    # Physical End:   130.0° (clockwise past 360) -> Logical 270.0°
    RAW_START_DEG = 220.0
    LOGICAL_MIN = 0.0
    LOGICAL_MAX = 270.0
    LOGICAL_RANGE = LOGICAL_MAX - LOGICAL_MIN
    
    # Inverted because the 1:2 gear mesh reverses rotation direction
    SERVO_DIRECTION = 1.0 

    def __init__(
        self,
        servo_bus=SERVO_I2C_BUS,
        as5600_bus=AS5600_I2C_BUS,
        servo_channel=CONTINUOUS_SERVO_CHANNEL,
        kp=0.04,
        ki=0.0,
        kd=0.01,
    ):
        self.servo_bus = servo_bus
        self.as5600_bus_id = as5600_bus
        self.servo_channel = servo_channel
        
        self.pwm_i2c = ExtendedI2C(self.servo_bus)
        self.pwm = PCA9685(self.pwm_i2c, address=self.PCA9685_ADDRESS)
        self.pwm.frequency = 50

        self.encoder_bus = smbus2.SMBus(self.as5600_bus_id)
        
        # PID setpoint = 0.0, feeding -error to drive positional error to zero
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=0.0, output_limits=(-1.0, 1.0))

    def read_logical_angle(self) -> float:
        """Calculates clockwise logical angle (0° to 270°) and clamps dead-zone readings."""
        try:
            raw = self.encoder_bus.read_i2c_block_data(
                self.AS5600_ADDRESS, self.AS5600_ANGLE_REGISTER, 2
            )
            raw_value = ((raw[0] << 8) | raw[1]) & 0x0FFF
            raw_deg = (raw_value / 4096.0) * 360.0
            
            # Convert to logical clockwise degrees from 220°
            logical_deg = (raw_deg - self.RAW_START_DEG) % 360.0
            
            # Clamp dead-zone readings (> 270.0°) to the nearest valid logical boundary
            if logical_deg > self.LOGICAL_MAX:
                if logical_deg > 315.0:  # Closer to 220° start (0.0°)
                    return self.LOGICAL_MIN
                else:  # Closer to 130° end (270.0°)
                    return self.LOGICAL_MAX
                    
            return logical_deg
        except Exception as e:
            print(f"Encoder read error: {e}")
            return self.LOGICAL_MIN

    def set_speed(self, output: float) -> None:
        speed = max(-1.0, min(1.0, float(output * self.SERVO_DIRECTION)))
        pulse = self.NEUTRAL_PULSE + int(speed * self.PULSE_SPAN)
        pulse = max(102, min(512, pulse))
        duty_cycle = int((pulse / 4096.0) * 65535)
        self.pwm.channels[self.servo_channel].duty_cycle = duty_cycle

    def normalized_to_degrees(self, normalized: float) -> float:
        normalized = max(0.0, min(1.0, float(normalized)))
        return self.LOGICAL_MIN + (normalized * self.LOGICAL_RANGE)

    def spin_to_target(self, target_normalized: float = 0.5, duration_sec: float = 6.0):
        target_deg = self.normalized_to_degrees(target_normalized)
        print(f"Starting test: Moving to target logical angle {target_deg:.1f}° (Normalized: {target_normalized})")
        
        start = time.monotonic()
        while time.monotonic() - start < duration_sec:
            current_deg = self.read_logical_angle()
            
            # Clean linear error since the logical range (0 to 270) has no internal wrapping
            error = target_deg - current_deg
            
            command = self.pid(-error)

            if abs(error) <= 2.0:  # 2-degree tolerance band
                command = 0.0
                self.set_speed(0.0)
                print("Target reached successfully!")
                break

            self.set_speed(command)
            print(f"Logical Current: {current_deg:.1f}° | Target: {target_deg:.1f}° | Error: {error:.1f}° | Speed: {command:.2f}")

            time.sleep(0.05)

        self.set_speed(0.0)
        print("Test complete. Motor stopped.")

    def close(self):
        try:
            self.set_speed(0.0)
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

if __name__ == "__main__":
    controller = ContinuousRotationServoControllerClean()
    try:
        # Test moving to midpoint (0.5 normalized = 135.0 logical degrees)
        controller.spin_to_target(target_normalized=0, duration_sec=6.0)
    finally:
        controller.close()