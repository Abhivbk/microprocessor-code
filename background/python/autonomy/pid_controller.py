import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import config


Point = Tuple[float, float]


@dataclass(frozen=True)
class ControlProposal:
    throttle: float
    brake: float
    steering: float
    target_point: Optional[Point]
    target_speed: float
    desired_steering: float
    heading_error: float
    speed_error: float
    lookahead_distance: float
    valid: bool
    reason: str
    timestamp_ms: int

    @classmethod
    def safe_stop(cls, reason: str) -> "ControlProposal":
        return cls(
            throttle=0.0,
            brake=1.0,
            steering=0.0,
            target_point=None,
            target_speed=0.0,
            desired_steering=0.0,
            heading_error=0.0,
            speed_error=0.0,
            lookahead_distance=0.0,
            valid=False,
            reason=reason,
            timestamp_ms=int(time.time() * 1000),
        )


class PID:
    """Small PID with derivative smoothing, anti-windup, and bounded output."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float,
        output_max: float,
        integral_limit: float,
        derivative_alpha: float,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = abs(integral_limit)
        self.derivative_alpha = min(1.0, max(0.0, derivative_alpha))
        self.reset()

    def update(self, error: float, dt: float) -> float:
        if not math.isfinite(error) or not math.isfinite(dt) or dt <= 0.0:
            return 0.0

        previous_integral = self.integral
        self.integral = _clamp(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        derivative = 0.0
        if self.previous_error is not None:
            raw_derivative = (error - self.previous_error) / dt
            derivative = (
                self.derivative_alpha * raw_derivative
                + (1.0 - self.derivative_alpha) * self.filtered_derivative
            )
        self.filtered_derivative = derivative
        self.previous_error = error

        unconstrained = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )
        output = _clamp(unconstrained, self.output_min, self.output_max)

        # Do not accumulate integral while saturation would push farther away.
        saturated_high = unconstrained > self.output_max and error > 0.0
        saturated_low = unconstrained < self.output_min and error < 0.0
        if saturated_high or saturated_low:
            self.integral = previous_integral

        return output

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0


class AutonomousPIDController:
    """Produce a safe dry-run FSDS control proposal from an EKF path."""

    def __init__(
        self,
        wheelbase: float = config.WHEELBASE_METERS,
        maximum_steering_degrees: float = config.MAXIMUM_STEERING_DEGREES,
        steering_sign: float = config.STEERING_SIGN,
        target_speed: float = config.TARGET_SPEED_METERS_PER_SECOND,
        base_lookahead: float = config.BASE_LOOKAHEAD_METERS,
        lookahead_speed_gain: float = config.LOOKAHEAD_SPEED_GAIN,
        minimum_lookahead: float = config.MINIMUM_LOOKAHEAD_METERS,
        maximum_lookahead: float = config.MAXIMUM_LOOKAHEAD_METERS,
        maximum_steering_rate: float = config.MAXIMUM_STEERING_RATE_PER_SECOND,
        maximum_dt: float = config.MAXIMUM_CONTROLLER_DT_SECONDS,
    ):
        self.wheelbase = wheelbase
        self.maximum_steering_radians = math.radians(maximum_steering_degrees)
        self.steering_sign = 1.0 if steering_sign >= 0.0 else -1.0
        self.target_speed = target_speed
        self.base_lookahead = base_lookahead
        self.lookahead_speed_gain = lookahead_speed_gain
        self.minimum_lookahead = minimum_lookahead
        self.maximum_lookahead = maximum_lookahead
        self.maximum_steering_rate = maximum_steering_rate
        self.maximum_dt = maximum_dt

        # Conservative initial gains adapted from the Pygame controller.
        self.steering_pid = PID(
            kp=config.STEERING_KP,
            ki=config.STEERING_KI,
            kd=config.STEERING_KD,
            output_min=-maximum_steering_rate,
            output_max=maximum_steering_rate,
            integral_limit=config.STEERING_INTEGRAL_LIMIT,
            derivative_alpha=config.STEERING_DERIVATIVE_ALPHA,
        )
        self.speed_pid = PID(
            kp=config.SPEED_KP,
            ki=config.SPEED_KI,
            kd=config.SPEED_KD,
            output_min=-1.0,
            output_max=1.0,
            integral_limit=config.SPEED_INTEGRAL_LIMIT,
            derivative_alpha=config.SPEED_DERIVATIVE_ALPHA,
        )

    def compute(
        self,
        car_pose: Sequence[float],
        current_speed: float,
        applied_steering: float,
        centerline: Sequence[Point],
        dt: float,
        path_is_valid: bool,
        imu_is_fresh: bool,
        actuator_is_fresh: bool,
        path_is_closed: bool = False,
    ) -> ControlProposal:
        invalid_reason = self._validate_inputs(
            car_pose=car_pose,
            current_speed=current_speed,
            applied_steering=applied_steering,
            centerline=centerline,
            dt=dt,
            path_is_valid=path_is_valid,
            imu_is_fresh=imu_is_fresh,
            actuator_is_fresh=actuator_is_fresh,
        )
        if invalid_reason:
            self.reset()
            return ControlProposal.safe_stop(invalid_reason)

        car_x, car_y, heading = map(float, car_pose[:3])
        speed = max(0.0, float(current_speed))
        applied = _clamp(float(applied_steering), -1.0, 1.0)
        path = [(float(x), float(y)) for x, y in centerline]
        if path_is_closed and len(path) > 2:
            if math.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]) < config.NUMERICAL_EPSILON:
                path = path[:-1]

        requested_lookahead = _clamp(
            self.base_lookahead + self.lookahead_speed_gain * speed,
            self.minimum_lookahead,
            self.maximum_lookahead,
        )
        target = self._find_lookahead_point(
            car_position=(car_x, car_y),
            centerline=path,
            lookahead_distance=requested_lookahead,
            closed=path_is_closed,
        )
        if target is None:
            self.reset()
            return ControlProposal.safe_stop("No forward lookahead target")

        dx = target[0] - car_x
        dy = target[1] - car_y
        actual_lookahead = math.hypot(dx, dy)
        if actual_lookahead < config.MINIMUM_TARGET_DISTANCE_METERS:
            self.reset()
            return ControlProposal.safe_stop("Lookahead target is too close")

        target_angle = math.atan2(dy, dx)
        heading_error = _normalize_angle(target_angle - heading)
        desired_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(heading_error),
            actual_lookahead,
        )
        desired_steering = self.steering_sign * _clamp(
            desired_angle / self.maximum_steering_radians,
            -1.0,
            1.0,
        )

        steering_error = desired_steering - applied
        steering_rate = self.steering_pid.update(steering_error, dt)
        steering_command = _clamp(
            applied + steering_rate * dt,
            -1.0,
            1.0,
        )

        speed_error = self.target_speed - speed
        longitudinal_command = self.speed_pid.update(speed_error, dt)
        if longitudinal_command >= 0.0:
            throttle = _clamp(longitudinal_command, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-longitudinal_command, 0.0, 1.0)

        return ControlProposal(
            throttle=throttle,
            brake=brake,
            steering=steering_command,
            target_point=target,
            target_speed=self.target_speed,
            desired_steering=desired_steering,
            heading_error=heading_error,
            speed_error=speed_error,
            lookahead_distance=actual_lookahead,
            valid=True,
            reason="",
            timestamp_ms=int(time.time() * 1000),
        )

    def reset(self):
        self.steering_pid.reset()
        self.speed_pid.reset()

    def _validate_inputs(
        self,
        car_pose: Sequence[float],
        current_speed: float,
        applied_steering: float,
        centerline: Sequence[Point],
        dt: float,
        path_is_valid: bool,
        imu_is_fresh: bool,
        actuator_is_fresh: bool,
    ) -> str:
        if not path_is_valid:
            return "Path is invalid"
        if len(car_pose) < 3 or not all(math.isfinite(float(v)) for v in car_pose[:3]):
            return "EKF pose is invalid"
        if not math.isfinite(float(current_speed)):
            return "Vehicle speed is invalid"
        if not math.isfinite(float(applied_steering)):
            return "Actuator steering is invalid"
        if not imu_is_fresh:
            return "IMU feedback is stale"
        if not actuator_is_fresh:
            return "Actuator feedback is stale"
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.maximum_dt:
            return "Controller time step is invalid"
        if len(centerline) < 2:
            return "Smoothed centerline is unavailable"
        for point in centerline:
            if len(point) < 2 or not all(math.isfinite(float(v)) for v in point[:2]):
                return "Centerline contains invalid points"
        return ""

    @staticmethod
    def _find_lookahead_point(
        car_position: Point,
        centerline: List[Point],
        lookahead_distance: float,
        closed: bool = False,
    ) -> Optional[Point]:
        if len(centerline) < 2:
            return None

        car_x, car_y = car_position
        closest_index = min(
            range(len(centerline)),
            key=lambda i: math.hypot(centerline[i][0] - car_x, centerline[i][1] - car_y),
        )

        remaining = lookahead_distance
        segment_count = len(centerline) if closed else len(centerline) - 1 - closest_index
        start = centerline[closest_index]
        for step in range(segment_count):
            index = (closest_index + step) % len(centerline)
            next_index = (index + 1) % len(centerline)
            if not closed and next_index <= index:
                break
            end = centerline[next_index]
            segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
            if segment_length >= remaining and segment_length > config.NUMERICAL_EPSILON:
                fraction = remaining / segment_length
                return (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            remaining -= segment_length
            start = end

        return centerline[closest_index] if closed else centerline[-1]


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
