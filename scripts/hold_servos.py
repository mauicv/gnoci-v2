"""Turns torque on with every servo holding where it is now, so nothing moves.

    uv run python scripts/hold_servos.py

Torque stays on after the script exits. Turn it off with `--off`.
"""
import sys

from rustypot import Sts3215PyController

IDS = [1, 2, 3, 4, 5, 6]

controller = Sts3215PyController(serial_port="/dev/serial0", baudrate=1_000_000, timeout=0.1)

if "--off" in sys.argv:
    controller.sync_write_torque_enable(IDS, [False] * len(IDS))
    print("Torque off")
    sys.exit()

positions = controller.sync_read_present_position(IDS)
controller.sync_write_goal_position(IDS, positions)
controller.sync_write_torque_enable(IDS, [True] * len(IDS))
print("Holding at:", dict(zip(IDS, [round(p, 3) for p in positions])))
