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

### Completed: EKF-driven Delaunay planning results

**Files changed:** `background/python/test.py`, `background/python/autonomy/__init__.py`, and `background/python/autonomy/path_planner.py`

- Added a planner that consumes confirmed EKF landmarks instead of generated cones.
- Added normalized blue/yellow cone filtering and ignored unknown/orange classes for boundary planning.
- Limited planning to a vehicle-relative forward region.
- Added guarded Delaunay triangulation and mixed-boundary triangle filtering.
- Added blue-yellow cross-edge midpoint calculation.
- Ordered the centerline from the EKF vehicle position in its heading direction without bridging disconnected gaps.
- Added natural open-path spline smoothing without forced loop closure.
- Added the requested `triangles`, `raw_centerline`, `smoothed_centerline`, and `is_valid` results.
- Limited replanning to 4 Hz.
- Drew triangles, raw centerline, and smoothed centerline in the existing EKF dashboard.
- Kept PID controls and minimum-curvature racing-line generation disconnected.

**Verification:** Static inspection only. No Python files, simulator, or tests were run.

### Completed: Timestamped actuator feedback node

**Files changed:** `foreground/python/actuator_state_node.py`, `foreground/src/main.rs`, and `background/python/test.py`

- Removed the separate Tkinter actuator window and converted the node to a lightweight headless worker.
- Added automatic FSDS reconnection and 20 Hz actuator feedback publishing.
- Preserved the existing `<Qfff` shared-memory format: timestamp, throttle, brake, and steering.
- Launched `actuator_state_node.py` automatically from the foreground Rust process.
- Preserved the actuator timestamp when the background dashboard reads feedback.
- Added a 250 ms freshness limit and exposed `actuator_feedback_fresh` for the future PID controller.
- Marked missing or old actuator feedback as waiting/stale in the dashboard.

**Verification:** Static inspection only. No Python, Rust, simulator, or tests were run.

### Completed: Dry-run pure-pursuit and PID control proposals

**Files changed:** `background/python/autonomy/pid_controller.py`, `background/python/autonomy/__init__.py`, and `background/python/test.py`

- Added a `ControlProposal` result containing throttle, brake, steering, target point, target speed, steering target, controller errors, validity, reason, and timestamp.
- Added open-path lookahead selection measured forward along the smoothed centerline.
- Added pure-pursuit steering based on the EKF position and IMU-corrected EKF heading.
- Added steering feedback control using fresh applied steering from `actuator_state_node.py`.
- Added a bounded steering-rate PID with derivative smoothing and integral anti-windup.
- Added a bounded speed PID that produces throttle for positive output and brake for negative output.
- Added configurable wheelbase, steering range/sign, target speed, lookahead, steering-rate, and time-step limits.
- Added strict validation of path, EKF pose, IMU freshness, actuator freshness, centerline points, and controller `dt`.
- Added a full-brake safe-stop proposal whenever controller inputs are invalid or stale.
- Added a dashboard dry-run panel and target-point visualization.
- Initially kept proposals separate from `self.desired`; the later guarded-live-autonomy change supersedes this dry-run-only stage.

**Verification:** Static inspection only. No Python, simulator, or tests were run.

### Completed: Guarded live autonomy, lap closure, and racing line

**Files changed:** dashboard, autonomy configuration, lap tracker, path planner, and PID controller modules.

- Added an exclusive manual/autonomous mode selector.
- Added a steering-sign verification flag that blocks live autonomy until explicitly confirmed.
- Connected valid, fresh PID proposals to the existing `self.desired` control output only in autonomous mode.
- Added automatic full braking for invalid paths, stale IMU/actuator/proposal data, or invalid controller inputs.
- Changed shared-memory control timestamps to represent the last real command update so the foreground watchdog can detect a frozen dashboard.
- Added safe initial full braking and a manual emergency-brake action that disables autonomy.
- Added EKF-based departure, travelled-distance, return-position, and return-heading lap checks.
- Added rejection of large single-step EKF jumps in lap-distance accumulation.
- Kept the first-lap centerline open and local.
- Added one-time background construction of an all-landmark closed centerline after confirmed lap completion.
- Added periodic closed-loop smoothing and one-time bounded minimum-curvature racing-line optimization.
- Added closed-path lookahead wrapping across the start/finish index.
- Switched the controller to the racing line only after successful loop closure and optimization.
- Added dashboard mode, safety, lap-distance, closure, and racing-line status plus racing-line drawing.

**Verification:** Static inspection only. No Python, Rust, simulator, or tests were run.

### Completed: Central autonomy configuration

**Files changed:** `background/python/autonomy/config.py`, planner/controller modules, and `background/python/test.py`

- Added one documented file for all autonomy tuning values and vehicle assumptions.
- Centralized sensor freshness limits, update rates, starting-view count, Delaunay thresholds, centerline settings, vehicle geometry, steering sign, target speed, lookahead settings, controller timing limits, and PID gains.
- Removed duplicated autonomy tuning values from the planner, PID controller, and dashboard.
- Kept protocol invariants and non-tunable mathematical values inside their implementation modules.

**Verification:** Static inspection only. No Python, simulator, or tests were run.

## Pending corrections

1. Add timestamps and closest-angle matching to camera-LiDAR fusion.
2. Add a safe manual/autonomous control selector and correct stale-command handling.
3. Stabilize shared-memory files so readers cannot observe partial writes.
4. Improve LiDAR cluster filtering and reject unknown camera cone classes.
5. Improve EKF covariance handling and update the outdated EKF tests.

## Approved Delaunay path-planning and PID merge corrections

1. Reuse the Delaunay triangulation, cross-track midpoint, centerline, pure-pursuit, and PID concepts from the separate project.
2. Remove Pygame, simulated vehicle physics, simulated visibility, and the prebuilt track generator from the production pipeline.
3. Build `Cone` objects from EKF landmarks instead of `track_generator.py`.
4. Normalize EKF labels such as `blue_cone`, `yellow_cone`, and `orange_cone` into the cone types expected by the planner.
5. Require enough valid cones, including both blue and yellow boundaries, before attempting Delaunay triangulation.
6. Start ordering centerline points from the waypoint nearest the EKF vehicle position and continue in the vehicle's heading direction; do not start from `cones[0]`.
7. Keep the discovered centerline open during the first lap. Do not connect its final point back to its first point while the map is incomplete.
8. Smooth the first-lap centerline using an open-path spline rather than a periodic/closed spline.
9. Add reliable start-line and lap-completion detection using the EKF pose and mapped start markers.
10. Close the centerline only after the car has completed one full lap and the start/end sections are confirmed to join safely.
11. Delay minimum-curvature racing-line generation until the first lap and global cone map are complete.
12. Compute the minimum-curvature racing line once after map completion, then reuse it instead of solving the optimization on every update.
13. Replan the first-lap centerline only when the EKF landmark map changes or at a limited planning frequency, not on every UI refresh.
14. Use `actuator_state_node.py` to provide actual applied steering feedback to the steering controller.
15. Launch `actuator_state_node.py` from the foreground process and verify its feedback timestamp before using it.
16. Replace the simulated steering state in the Pygame controller with fresh actuator feedback, with the last safe command as a temporary fallback.
17. Convert the controller's steering angle into FSDS's normalized `[-1, 1]` steering command and verify the steering sign convention.
18. Change the speed controller to produce both throttle and brake. Do not depend on simulated friction to reduce speed.
19. Add PID integral limits, reset behavior, output limits, and safe handling for invalid or unusually large `dt` values.
20. Add a manual/autonomous mode selector so only one source can write the desired control command at a time.
21. Apply full braking when the EKF pose, planned path, actuator feedback, or autonomous command becomes stale or invalid.
22. First integrate `EKF map -> Delaunay -> centerline drawing` without autonomous control. Connect PID outputs to FSDS only after the displayed centerline is verified.
