#!/usr/bin/env python3
"""Read the yaw encoder: raw AS5600 angle, logical angle (0-325), and 0-1 position.

    python read_encoder.py            # one reading
    python read_encoder.py watch 0.2  # keep reading every 0.2 s (Ctrl+C to stop)

Never drives the servo; only reads the encoder.
"""

import sys
import time

from hardware.config import AS5600_I2C_BUS
from hardware.servo_controller import ContinuousRotationServoController


def describe(cont: ContinuousRotationServoController) -> str:
    raw = cont.try_read_raw_angle()
    if raw is None:
        return "encoder read failed"
    logical = cont.raw_to_logical(raw)
    in_dead_zone = (raw - cont.RAW_START_DEG) % 360.0 > cont.LOGICAL_MAX
    return (
        f"raw {raw:7.2f}° | logical {logical:6.2f}° | position {cont.degrees_to_normalized(logical):.3f}"
        + ("  [dead zone, clamped]" if in_dead_zone else "")
    )


def main():
    cont = ContinuousRotationServoController()
    if not cont.encoder_available:
        print(f"AS5600 encoder not found on bus {AS5600_I2C_BUS}")
        sys.exit(1)

    print(
        f"Logical 0° = raw {cont.RAW_START_DEG:.1f}°, logical {cont.LOGICAL_MAX:.0f}° = raw "
        f"{(cont.RAW_START_DEG + cont.LOGICAL_MAX) % 360:.1f}° (clockwise)"
    )
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        print(f"Watching encoder every {interval}s. Press Ctrl+C to stop.")
        try:
            while True:
                print(describe(cont))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print(describe(cont))


if __name__ == "__main__":
    main()
