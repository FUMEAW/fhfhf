import statistics

from hardware.config import (
    LED_COUNT,
    NEOPIXEL_PIN,
    TCA9548A_I2C_BUS,
    ULTRASONIC_ECHO,
    ULTRASONIC_TRIGGER,
)

empty_height = 287

_pixels = None
_ultra_sensor = None
_multiplex = None
_tof_sensors = {}


def _get_pixels():
    global _pixels
    if _pixels is None:
        import board
        import neopixel

        pin = getattr(board, f"D{NEOPIXEL_PIN}")
        _pixels = neopixel.NeoPixel(pin, LED_COUNT, brightness=0.5, auto_write=False)
    return _pixels


def _get_ultra():
    global _ultra_sensor
    if _ultra_sensor is None:
        from gpiozero import DistanceSensor

        _ultra_sensor = DistanceSensor(
            echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIGGER, max_distance=4.0
        )
    return _ultra_sensor


def _get_mux():
    global _multiplex
    if _multiplex is None:
        import adafruit_tca9548a
        from adafruit_extended_bus import ExtendedI2C

        _multiplex = adafruit_tca9548a.TCA9548A(ExtendedI2C(TCA9548A_I2C_BUS))
    return _multiplex


def tof_measure_value(channel):
    import adafruit_vl53l0x

    mux = _get_mux()
    if channel not in _tof_sensors:
        tof = adafruit_vl53l0x.VL53L0X(mux[channel])
        tof.measurement_timing_budget = 200000
        _tof_sensors[channel] = tof
    return 1 - ((_tof_sensors[channel].range - 133) / empty_height)


def tof_value(channel):
    rds = [tof_measure_value(channel) for _ in range(4)]
    return statistics.median(rds)


def set_ultras_leds(color):
    pixels = _get_pixels()
    for i in range(LED_COUNT):
        pixels[i] = color[i]
    pixels.show()


def ultrasonic_val():
    distance_mm = _get_ultra().distance * 1000.0
    return max(0.0, min(1.0, (empty_height - distance_mm) / empty_height))
