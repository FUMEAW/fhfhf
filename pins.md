### Pin Assignment
**This is to map out how pins are assigned, and what they are assigned to**

##### Servo Driver (PCA9685)
Hardware I2C `/dev/i2c-1` (GPIO 2 SDA / GPIO 3 SCL)
- Channels 0-2: platform tilt
- Channel 3: continuous-rotation yaw servo (replaces the DC motor driver)
- Channels 4-7: slot flaps / doors

##### Ultrasonic Sensor
G18 - RGB, G24 - Echo, G23 - Trigger

##### TOF Sensor Multiplexer (TCA9548A)
`/dev/i2c-5`

##### AS5600 Encoder
`/dev/i2c-4`  (GPIO 5 SDA / GPIO 6 SCL)

##### Retired: Motor Driver
GPIO 17 / 27 / 22 are unused. Yaw is the continuous servo on PCA9685 channel 3.

##### Stuff to paste into config.txt
###### Software I2C for the AS5600 (`/dev/i2c-4`)
dtoverlay=i2c-gpio,i2c_gpio_sda=5,i2c_gpio_scl=6,i2c_gpio_delay_us=5

###### Software I2C for the TCA9548A (`/dev/i2c-5`)
dtoverlay=i2c-gpio,i2c_gpio_sda=15,i2c_gpio_scl=17,i2c_gpio_delay_us=5

Above to be pasted in: `/boot/firmware/config.txt`
