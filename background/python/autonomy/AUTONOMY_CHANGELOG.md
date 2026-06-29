# Autonomy Change Log

## Completed

- Fixed zero-heading EKF initialization and one-time stationary startup mapping.
- Added EKF-driven Delaunay triangles, raw centerline, and open smoothed centerline.
- Added a headless, timestamped actuator-feedback node and automatic foreground launch.
- Added pure-pursuit steering plus bounded steering and speed PID controllers.
- Added stale/invalid-data braking and truthful control timestamps for the watchdog.
- Added exclusive manual/autonomous command selection and emergency manual braking.
- Added EKF-based lap detection, post-lap loop closure, and one-time minimum-curvature racing-line generation.
- Centralized autonomy tuning in `config.py`.

## Current safety state

Autonomous commands reach FSDS only when autonomous mode is enabled and all path, IMU, actuator, and controller checks are valid. Otherwise, the system commands full braking.

Live autonomy remains blocked until these values are confirmed in `config.py`:

```python
STEERING_SIGN = 1.0  # Change to -1.0 if FSDS steering is reversed.
STEERING_SIGN_VERIFIED = True
```

## Verification

- Static inspection and diff checks completed.
- Simulator and runtime tests have not been performed.
