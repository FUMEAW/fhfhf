"""Bin fill-level sensors.

Three VL53L0X time-of-flight sensors (recycling, compost, landfill) sit behind
a TCA9548A multiplexer, one per channel, because they all share address 0x29.
The battery bin uses an HC-SR04 ultrasonic sensor on GPIO echo/trigger.

Every sensor looks down into its bin and reports a distance. Fill is
    (empty_mm - distance) / (empty_mm - full_mm), clamped to 0-1,
with per-bin distances in hardware.config.BIN_SENSOR_RANGE_MM.

BinCapacityMonitor polls all four in a background thread so the sort loop
never waits on the sensors. Test the sensors on their own with:
    python -m hardware.tof_ultrasonic_multiplexer
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Dict, Optional

from hardware.config import (
    BIN_FULL_THRESHOLD,
    BIN_SENSOR_RANGE_MM,
    LED_COUNT,
    NEOPIXEL_PIN,
    TCA9548A_I2C_BUS,
    TOF_MUX_CHANNELS,
    ULTRASONIC_BIN,
    ULTRASONIC_ECHO,
    ULTRASONIC_TRIGGER,
)

# VL53L0X reports 8190/8191 when nothing is in range or the reading failed.
TOF_INVALID_MM = 8000
TOF_SAMPLES = 4
ULTRASONIC_MAX_MM = 4000.0

_pixels = None
_i2c = None
_multiplex = None
_ultra_sensor = None
_tof_sensors = {}


# ---------------------------------------------------------------------------
# Bus and device setup (lazy, so importing this module never touches hardware)
# ---------------------------------------------------------------------------

def _get_pixels():
    global _pixels
    if _pixels is None:
        import board
        import neopixel

        pin = getattr(board, f"D{NEOPIXEL_PIN}")
        _pixels = neopixel.NeoPixel(pin, LED_COUNT, brightness=0.5, auto_write=False)
    return _pixels


def _get_i2c():
    global _i2c
    if _i2c is None:
        from adafruit_extended_bus import ExtendedI2C

        _i2c = ExtendedI2C(TCA9548A_I2C_BUS)
    return _i2c


def _get_mux():
    global _multiplex
    if _multiplex is None:
        import adafruit_tca9548a

        _multiplex = adafruit_tca9548a.TCA9548A(_get_i2c())
    return _multiplex


def _get_tof(channel: int):
    if channel not in _tof_sensors:
        import adafruit_vl53l0x

        tof = adafruit_vl53l0x.VL53L0X(_get_mux()[channel])
        tof.measurement_timing_budget = 200000
        _tof_sensors[channel] = tof
    return _tof_sensors[channel]


def _get_ultra():
    global _ultra_sensor
    if _ultra_sensor is None:
        from gpiozero import DistanceSensor

        _ultra_sensor = DistanceSensor(
            echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIGGER, max_distance=4.0
        )
    return _ultra_sensor


def _reset_device(bin_name: str) -> None:
    """Forget a sensor after an error so the next read re-initialises it."""
    global _ultra_sensor
    if bin_name == ULTRASONIC_BIN:
        _ultra_sensor = None
    elif bin_name in TOF_MUX_CHANNELS:
        _tof_sensors.pop(TOF_MUX_CHANNELS[bin_name], None)


# ---------------------------------------------------------------------------
# Raw distances
# ---------------------------------------------------------------------------

def tof_distance_mm(channel: int) -> Optional[float]:
    """Median of TOF_SAMPLES VL53L0X readings on a mux channel, None if all invalid."""
    tof = _get_tof(channel)
    readings = [tof.range for _ in range(TOF_SAMPLES)]
    valid = [r for r in readings if 0 < r < TOF_INVALID_MM]
    return float(statistics.median(valid)) if valid else None


def ultrasonic_distance_mm() -> Optional[float]:
    # gpiozero reports max_distance (4 m) when no echo comes back.
    distance = _get_ultra().distance * 1000.0
    return distance if 0 < distance < ULTRASONIC_MAX_MM else None


def bin_distance_mm(bin_name: str) -> Optional[float]:
    if bin_name == ULTRASONIC_BIN:
        return ultrasonic_distance_mm()
    return tof_distance_mm(TOF_MUX_CHANNELS[bin_name])


def fill_from_distance(bin_name: str, distance_mm: float) -> float:
    rng = BIN_SENSOR_RANGE_MM[bin_name]
    span = rng["empty_mm"] - rng["full_mm"]
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (rng["empty_mm"] - distance_mm) / span))


# Older helpers, kept for existing callers. Both return fill 0-1.

def tof_value(channel: int) -> Optional[float]:
    bin_name = next((b for b, c in TOF_MUX_CHANNELS.items() if c == channel), None)
    distance = tof_distance_mm(channel)
    if bin_name is None or distance is None:
        return None
    return fill_from_distance(bin_name, distance)


def ultrasonic_val() -> Optional[float]:
    distance = ultrasonic_distance_mm()
    return None if distance is None else fill_from_distance(ULTRASONIC_BIN, distance)


def set_ultras_leds(color):
    pixels = _get_pixels()
    for i in range(LED_COUNT):
        pixels[i] = color[i]
    pixels.show()


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------

class BinCapacityMonitor:
    """Polls every bin sensor in a background thread and keeps the latest fill."""

    def __init__(
        self,
        poll_interval: float = 5.0,
        full_threshold: float = BIN_FULL_THRESHOLD,
    ):
        self.poll_interval = float(poll_interval)
        self.full_threshold = float(full_threshold)
        self.bins = list(TOF_MUX_CHANNELS) + [ULTRASONIC_BIN]
        self._lock = threading.Lock()
        self._readings: Dict[str, Dict] = {
            name: {"fill": None, "distance_mm": None, "ok": False, "error": None, "time": None}
            for name in self.bins
        }
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def read_bin(self, bin_name: str) -> Dict:
        """Read one sensor now and store the result."""
        try:
            distance = bin_distance_mm(bin_name)
            if distance is None:
                reading = {"fill": None, "distance_mm": None, "ok": False,
                           "error": "out of range"}
            else:
                reading = {"fill": round(fill_from_distance(bin_name, distance), 3),
                           "distance_mm": round(distance, 1), "ok": True, "error": None}
        except Exception as exc:
            _reset_device(bin_name)
            reading = {"fill": None, "distance_mm": None, "ok": False, "error": str(exc)}
        reading["time"] = time.time()
        with self._lock:
            self._readings[bin_name] = reading
        return reading

    def read_all(self) -> Dict[str, Dict]:
        for name in self.bins:
            self.read_bin(name)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Dict]:
        """Latest reading per bin, with a "full" flag added."""
        with self._lock:
            out = {name: dict(r) for name, r in self._readings.items()}
        for r in out.values():
            r["full"] = r["fill"] is not None and r["fill"] >= self.full_threshold
        return out

    def fill(self, bin_name: str) -> Optional[float]:
        with self._lock:
            reading = self._readings.get(bin_name)
        return None if reading is None else reading["fill"]

    def is_full(self, bin_name: str) -> bool:
        """True only when the sensor positively reads full. Unknown is not full."""
        fill = self.fill(bin_name)
        return fill is not None and fill >= self.full_threshold

    @property
    def available(self) -> bool:
        with self._lock:
            return any(r["ok"] for r in self._readings.values())

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            for name in self.bins:
                if self._stop.is_set():
                    return
                self.read_bin(name)
            self._stop.wait(self.poll_interval)


if __name__ == "__main__":
    monitor = BinCapacityMonitor()
    print(f"I2C bus {TCA9548A_I2C_BUS}: ToF channels {TOF_MUX_CHANNELS}, "
          f"ultrasonic (trigger GPIO {ULTRASONIC_TRIGGER}, echo GPIO {ULTRASONIC_ECHO}) "
          f"on {ULTRASONIC_BIN}. Ctrl+C to stop.")
    try:
        while True:
            for name, r in monitor.read_all().items():
                if r["ok"]:
                    flag = "  FULL" if r["full"] else ""
                    print(f"  {name:<10} {r['distance_mm']:7.1f} mm  {r['fill'] * 100:5.1f}%{flag}")
                else:
                    print(f"  {name:<10}  -- {r['error']}")
            print()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
