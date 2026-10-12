"""Main AutoSort orchestration layer.

This module wires together the vision pipeline, platform motion, and rotation
control into a single clean flow:

    classify -> rotate to bin -> tilt to slot -> open door -> eject -> close door -> level -> home
"""

from __future__ import annotations

from autosort.planner import AutoSortSystem


if __name__ == "__main__":
    system = AutoSortSystem(camera_index=None)
    try:
        system.start()
        system.run_loop(max_cycles=200)
    except KeyboardInterrupt:
        print("AutoSort loop interrupted.")
    finally:
        system.stop()


