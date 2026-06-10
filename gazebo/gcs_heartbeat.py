#!/usr/bin/env python3
"""
Connects to PX4 via MAVLink, sets battery/arming params, and sends
continuous GCS heartbeats so PX4 doesn't block arming.

Run this AFTER "Ready for takeoff!" appears in the PX4 terminal.
Keep it running while flying.

Reference: https://docs.px4.io/main/en/simulation/failsafes.html
"""

from pymavlink import mavutil
import time


def main():
    print("Connecting to PX4 on UDP 14540...")
    conn = mavutil.mavlink_connection('udpin:localhost:14540', source_system=255)
    conn.wait_heartbeat(timeout=10)
    print(f"Connected to PX4 (system {conn.target_system})")

    # INT32 params
    int_params = [
        (b'COM_LOW_BAT_ACT', 0),   # warning only, no forced landing on low battery
        (b'NAV_DLL_ACT',     0),   # disable data-link loss action
        (b'COM_RCL_EXCEPT',  7),   # allow offboard without RC
    ]
    # FLOAT params
    float_params = [
        (b'BAT_LOW_THR',     0.0),
        (b'BAT_CRIT_THR',    0.0),
        (b'BAT_EMERGEN_THR', 0.0),
        (b'COM_ARM_BAT_MIN', 0.0),
    ]

    for name, val in int_params:
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            name, float(val), mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        time.sleep(0.2)
        print(f"  Set {name.decode()} = {int(val)}")

    for name, val in float_params:
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            name, float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.2)
        print(f"  Set {name.decode()} = {val}")

    print("Sending GCS heartbeats — keep this running while flying...")
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
