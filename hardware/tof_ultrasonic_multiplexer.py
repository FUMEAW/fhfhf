import board
import adafruit_tca9548a
import time
import adafruit_vl53l0x
import neopixel
from gpiozero import DistanceSensor
import busio
import smbus2
from adafruit_extended_bus import ExtendedI2C
import statistics

# Initialize the built-in NeoPixel RGB LEDs (6 total, 3 in each barrel)
# GPIO 18 (board.D18) is ideal for hardware PWM control
LED_COUNT = 6
pixels = neopixel.NeoPixel(board.D18, LED_COUNT, brightness=0.5, auto_write=False)
ultra_sensor = DistanceSensor(echo=24, trigger=23, max_distance=4.0)

# CHANGE THIS to the exact digit you found at the end of 'ls /dev/i2c*' to use bitbang
# /dev/i2c-13
MY_ASSIGNED_BUS = 5  
i2c = ExtendedI2C(MY_ASSIGNED_BUS)
multiplex = adafruit_tca9548a.TCA9548A(i2c)

# Tune this value after all the electronics are put together
empty_height = 287

# Used to determine bin fullness for all bins but battery bin
def tof_measure_value(channel):
    tof = adafruit_vl53l0x.VL53L0X(multiplex[channel])
    tof.measurement_timing_budget = 200000
    return 1 - ((tof.range - 133)/empty_height)
    # Returns a fraction from 0.0 (empty) to 1.0 (full)

def tof_value(channel):
    rds = []
    for i in range(0,4):
        rds.append(tof_measure_value(channel))
    return statistics.median(rds)

def set_ultras_leds(color):
    for i in range(LED_COUNT):
        pixels[i] = color[i]
    pixels.show()

# Used to determine bin fullness for all bins for battery bin
def ultrasonic_val():
    distance_mm = ultra_sensor.distance * 1000.0
    return max(0.0, min(1.0, (empty_height - distance_mm) / empty_height))
