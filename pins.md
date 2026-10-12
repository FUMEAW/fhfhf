### Pin Assignment
**This is to map out how pins are assigned, and what they are assigned to**

i2c pins -- Servo Driver

##### Ultrasonic Sensor
G18 - RGB, G24 - Echo, G23 - Trigger

##### TOF Sensor Multiplexer
G10 - SDA, G11 - SCL

##### AS5600 Encoder Pins
G5 - SDA, G6 - SCL

##### Motor Driver Pins
G17 - IN1, G27 - IN2, G22 - EN

##### Stuff to paste into config tile
###### Bus 1: For the TCA9548A Multiplexer (Physical Pins 10 & 11)
dtoverlay=i2c-gpio,i2c_gpio_sda=15,i2c_gpio_scl=17,i2c_gpio_delay_us=5
PASTED IT IS /dev/i2c-4 for the bus

###### Bus 2: For the AS5600 Encoder (Physical Pins 29 & 31)
dtoverlay=i2c-gpio,i2c_gpio_sda=5,i2c_gpio_scl=6,i2c_gpio_delay_us=5
PASTED IT IS /dev/i2c-5 for the bus

Above to be pasted in: "/boot/firmware/config.txt"