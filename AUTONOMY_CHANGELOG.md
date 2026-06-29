# Autonomy Integration Change Log

This file tracks code changes made while preparing the project for autonomous path planning and PID control.

## 2026-06-29

### Completed: EKF heading initialization and starting-view mapping

**File changed:** `background/python/test.py`

- Accepted the valid zero-degree IMU quaternion `(0, 0, 0, 1)` instead of treating it as missing heading data.
- Stored and checked the IMU timestamp before using orientation data.
- Validated and normalized the complete quaternion.
- Calculated yaw using all four quaternion components.
- Added one-time mapping of cones visible from the initial stationary position.
- Used two stationary EKF updates so provisional cones can be confirmed as landmarks.
- Prevented repeated stationary updates after the starting view has been mapped.
- Preserved continuous EKF cone mapping while speed is greater than `0.1 m/s`.
- Prevented later stops from being mistaken for the initial stationary view.

**Verification:** Static diff and formatting checks only. No programs, simulator, or tests were run.

## Pending corrections

1. Add timestamps and closest-angle matching to camera-LiDAR fusion.
2. Add a safe manual/autonomous control selector and correct stale-command handling.
3. Stabilize shared-memory files so readers cannot observe partial writes.
4. Improve LiDAR cluster filtering and reject unknown camera cone classes.
5. Improve EKF covariance handling and update the outdated EKF tests.

