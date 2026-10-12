#!/usr/bin/env python3
"""Fast platform spinner - lightweight, no heavy dependencies."""

import sys
import argparse
from hardware.config import AS5600_I2C_BUS, CONTINUOUS_SERVO_CHANNEL, SERVO_I2C_BUS
from hardware.servo_controller import ContinuousRotationServoController


def main():
    parser = argparse.ArgumentParser(
        description="Spin the platform using 0-1 normalized position (0=268.33°, 1=286.70°)",
        prog="python spin_platform.py"
    )
    parser.add_argument(
        "position",
        type=float,
        nargs="?",
        default=0.5,
        help="Target position 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=2.0,
        help="Spin duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "-p", "--print-samples",
        action="store_true",
        help="Print sample data during spin"
    )
    
    args = parser.parse_args()
    position = max(0.0, min(1.0, args.position))
    duration = max(0.1, args.duration)
    
    print(f"Initializing platform controller...")
    cont = ContinuousRotationServoController(
        servo_bus=SERVO_I2C_BUS,
        as5600_bus=AS5600_I2C_BUS,
        servo_channel=CONTINUOUS_SERVO_CHANNEL,
    )
    
    if not cont.servo_available:
        print(f"ERROR: PCA9685 servo controller not found on bus {SERVO_I2C_BUS}")
        sys.exit(1)
    
    print(f"Refreshing encoder connection...")
    if not cont.refresh_encoder():
        print(f"ERROR: AS5600 encoder not found on bus {AS5600_I2C_BUS}")
        sys.exit(1)
    
    print(f"\nSpinning to position {position:.2f} (0-1) = {cont.normalized_to_degrees(position):.2f}°")
    print(f"Duration: {duration:.1f}s")
    print()
    
    result = cont.spin_to_target(target_normalized=position, duration_sec=duration)
    
    print(f"\n✓ Spin complete!")
    print(f"  Target position: {result['target_normalized']:.2f} (0-1)")
    print(f"  Target angle: {result['target_deg']:.2f}°")
    print(f"  Final angle: {result['final_angle_deg']:.2f}°")
    print(f"  Final position: {result['final_normalized']:.3f} (0-1)")
    
    if args.print_samples and result['samples']:
        print(f"\nSamples ({len(result['samples'])} total):")
        for i, sample in enumerate(result['samples']):
            print(f"  {i+1:2d}: t={sample['time']:5.2f}s | angle={sample['angle_deg']:7.2f}° | " +
                  f"norm={sample['normalized_pos']:.3f} | error={sample['error_deg']:6.2f}° | " +
                  f"cmd={sample['command']:6.3f}")


if __name__ == "__main__":
    main()
