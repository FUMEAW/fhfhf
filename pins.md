### Pin Assignment
**This is to map out how pins are assigned, and what they are assigned to**

##### Servo Driver (PCA9685)
Hardware I2C `/dev/i2c-1` (GPIO 2 SDA / GPIO 3 SCL)
- Channels 0-2: platform tilt
- Channel 3: continuous-rotation yaw servo (replaces the DC motor driver)
- Channels 4-7: slot flaps / doors

##### NeoPixel LEDs
G18

##### Bin Fill Sensors
- TCA9548A multiplexer (0x70) on `/dev/i2c-5` (software I2C, GPIO 23/24)
  - Channel 0: VL53L0X, recycling bin
  - Channel 1: VL53L0X, compost bin
  - Channel 2: VL53L0X, landfill bin
- HC-SR04 ultrasonic, battery bin: G16 - Trigger, G20 - Echo
  - Echo is 5 V: put a voltage divider (e.g. 1k / 2k) between it and G20

Channel/bin mapping and empty/full distances live in `hardware/config.py`.
Test with `python -m hardware.tof_ultrasonic_multiplexer`.

##### AS5600 Encoder
`/dev/i2c-4`  (GPIO 5 SDA / GPIO 6 SCL)

##### Retired: Motor Driver
GPIO 17 / 27 / 22 are unused. Yaw is the continuous servo on PCA9685 channel 3.

##### Stuff to paste into config.txt
###### Software I2C for the AS5600 (`/dev/i2c-4`)
dtoverlay=i2c-gpio,i2c_gpio_sda=5,i2c_gpio_scl=6,i2c_gpio_delay_us=5

###### Software I2C for the TCA9548A (`/dev/i2c-5`)
dtoverlay=i2c-gpio,i2c_gpio_sda=23,i2c_gpio_scl=24,i2c_gpio_delay_us=5

Above to be pasted in: `/boot/firmware/config.txt`
