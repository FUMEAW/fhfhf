"""Shared Raspberry Pi bus, address, and PCA9685 channel map.

  /dev/i2c-1  PCA9685 servo driver (0x40)
  /dev/i2c-4  AS5600 encoder (0x36)
  /dev/i2c-5  TCA9548A ToF multiplexer

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
SLOT_FLAP_CHANNELS = {1: 7, 2: 5, 3: 4, 4: 6}

ULTRASONIC_ECHO = 24
ULTRASONIC_TRIGGER = 23
NEOPIXEL_PIN = 18
LED_COUNT = 6
