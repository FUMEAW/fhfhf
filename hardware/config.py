"""Shared Raspberry Pi bus, address, and PCA9685 channel map.

  /dev/i2c-1  PCA9685 servo driver (0x40)
  /dev/i2c-4  AS5600 encoder (0x36)
  /dev/i2c-5  TCA9548A ToF multiplexer (0x70) + I2C ultrasonic (0x57)

The DC motor driver (GPIO 17/27/22) is retired. Platform yaw is a
continuous-rotation servo on PCA9685 channel 3 with AS5600 feedback.
"""

SERVO_I2C_BUS = 1
PCA9685_ADDRESS = 0x40

TCA9548A_I2C_BUS = 5
AS5600_I2C_BUS = 4
AS5600_ADDRESS = 0x36
AS5600_ANGLE_REGISTER = 0x0E

TILT_CHANNELS = (0, 1, 2)
CONTINUOUS_SERVO_CHANNEL = 3
SLOT_FLAP_CHANNELS = {1: 4, 2: 6, 3: 7, 4: 5}  # slot -> PCA9685 channel of the flap physically at that slot

# Bin fill sensors, all on TCA9548A_I2C_BUS. The three VL53L0X share address
# 0x29, so each sits on its own multiplexer channel. The battery bin uses an
# ultrasonic sensor directly on the same bus (not behind the multiplexer).
TOF_MUX_CHANNELS = {"recycling": 0, "compost": 1, "landfill": 2}  # bin -> TCA9548A channel
ULTRASONIC_BIN = "battery"
ULTRASONIC_MODE = "i2c"          # "i2c" (RCWL-9620 / RCWL-1601 style) or "gpio" (HC-SR04)
ULTRASONIC_I2C_ADDRESS = 0x57

# Sensor-to-surface distance (mm) when each bin is empty and when it is full.
# fill = (empty_mm - reading) / (empty_mm - full_mm), clamped to 0-1.
BIN_SENSOR_RANGE_MM = {
    "recycling": {"empty_mm": 420.0, "full_mm": 133.0},
    "compost": {"empty_mm": 420.0, "full_mm": 133.0},
    "landfill": {"empty_mm": 420.0, "full_mm": 133.0},
    "battery": {"empty_mm": 287.0, "full_mm": 0.0},
}
BIN_FULL_THRESHOLD = 0.9         # fill at or above this = bin full, stop sorting into it

ULTRASONIC_ECHO = 24             # only used when ULTRASONIC_MODE == "gpio"
ULTRASONIC_TRIGGER = 23
NEOPIXEL_PIN = 18
LED_COUNT = 6
