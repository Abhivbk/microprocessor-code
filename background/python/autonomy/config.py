"""Single source of truth for autonomy tuning values.

Edit values in this file when calibrating the EKF-to-path-to-control pipeline.
Keep algorithm logic in the planner and controller modules.
"""

# ---------------------------------------------------------------------------
# Sensor freshness and update rates
# ---------------------------------------------------------------------------

IMU_STALE_MS = 250
ACTUATOR_STALE_MS = 250
CONTROL_SOURCE_STALE_MS = 250
PATH_PLAN_INTERVAL_SECONDS = 0.25  # 4 Hz
STARTING_VIEW_REQUIRED_UPDATES = 2

# ---------------------------------------------------------------------------
# EKF landmark and Delaunay path-planning configuration
# ---------------------------------------------------------------------------

TRACK_WIDTH_METERS = 3.5
MIN_LANDMARK_HIT_COUNT = 2
MIN_TOTAL_BOUNDARY_CONES = 4
MIN_CONES_PER_BOUNDARY_COLOR = 2

PLANNING_DISTANCE_METERS = 35.0
PLANNING_BEHIND_DISTANCE_METERS = 3.0
PLANNING_LATERAL_LIMIT_METERS = 15.0

MIN_CROSS_TRACK_WIDTH_METERS = 1.0
MAX_CROSS_TRACK_WIDTH_METERS = 7.0
MAX_TRIANGLE_EDGE_METERS = TRACK_WIDTH_METERS * 2.5
MAX_CENTERLINE_GAP_METERS = 8.0
CONE_DUPLICATE_DISTANCE_METERS = 0.25

CENTERLINE_SMOOTHING_POINTS = 80
CLOSED_CENTERLINE_SMOOTHING_POINTS = 200
CENTERLINE_START_MIN_FORWARD_METERS = -1.0
MIN_CENTERLINE_DIRECTION_ALIGNMENT = -0.15
MIN_VALID_CENTERLINE_POINTS = 2

# ---------------------------------------------------------------------------
# Vehicle and pure-pursuit assumptions
# ---------------------------------------------------------------------------

WHEELBASE_METERS = 1.5
MAXIMUM_STEERING_DEGREES = 30.0

# +1 keeps the mathematical steering direction; -1 reverses it for FSDS.
# Existing keyboard controls suggest FSDS may require -1. Keep proposals in
# dry-run mode until this value is verified on the actual simulator.
STEERING_SIGN = 1.0
STEERING_SIGN_VERIFIED = True

TARGET_SPEED_METERS_PER_SECOND = 2.0
BASE_LOOKAHEAD_METERS = 3.0
LOOKAHEAD_SPEED_GAIN = 0.4
MINIMUM_LOOKAHEAD_METERS = 3.0
MAXIMUM_LOOKAHEAD_METERS = 6.0
MINIMUM_TARGET_DISTANCE_METERS = 0.25

MAXIMUM_STEERING_RATE_PER_SECOND = 2.0
MAXIMUM_CONTROLLER_DT_SECONDS = 0.20

# ---------------------------------------------------------------------------
# Lap completion and racing-line generation
# ---------------------------------------------------------------------------

LAP_START_RADIUS_METERS = 2.0
LAP_DEPARTURE_RADIUS_METERS = 4.0
MINIMUM_LAP_DISTANCE_METERS = 30.0
MAXIMUM_LAP_RETURN_HEADING_ERROR_DEGREES = 60.0

RACING_LINE_SAFETY_MARGIN_METERS = 0.5
RACING_LINE_OPTIMIZER_MAX_ITERATIONS = 2000
RACING_LINE_OPTIMIZER_FTOL = 1e-12
RACING_LINE_OPTIMIZER_GTOL = 1e-8
RACING_LINE_REGULARIZATION = 1e-6
MAX_EKF_LAP_STEP_METERS = 8.0

# ---------------------------------------------------------------------------
# Steering PID gains and limits
# ---------------------------------------------------------------------------

STEERING_KP = 1.2
STEERING_KI = 0.0
STEERING_KD = 0.1
STEERING_INTEGRAL_LIMIT = 0.5
STEERING_DERIVATIVE_ALPHA = 0.25

# ---------------------------------------------------------------------------
# Speed PID gains and limits
# ---------------------------------------------------------------------------

SPEED_KP = 1.5
SPEED_KI = 0.1
SPEED_KD = 0.05
SPEED_INTEGRAL_LIMIT = 1.0
SPEED_DERIVATIVE_ALPHA = 0.25

# Numerical threshold used to avoid division by zero in path calculations.
NUMERICAL_EPSILON = 1e-9
