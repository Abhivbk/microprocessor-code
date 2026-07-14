import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from . import config


Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class LapStatus:
    lap_complete: bool
    just_completed: bool
    departed_start: bool
    distance_travelled: float
    distance_from_start: float
    reason: str = ""


class LapTracker:
    """Detect one completed lap from EKF pose without using simulated physics."""

    def __init__(self):
        self.start_pose: Optional[Pose] = None
        self.last_position: Optional[Tuple[float, float]] = None
        self.departed_start = False
        self.lap_complete = False
        self.distance_travelled = 0.0

    def reset(self, pose: Optional[Sequence[float]] = None):
        self.start_pose = None
        self.last_position = None
        self.departed_start = False
        self.lap_complete = False
        self.distance_travelled = 0.0
        if pose is not None and self._pose_is_valid(pose):
            self.start_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
            self.last_position = (float(pose[0]), float(pose[1]))

    def update(self, pose: Sequence[float]) -> LapStatus:
        if not self._pose_is_valid(pose):
            return self._status(reason="EKF pose is invalid")

        x, y, heading = float(pose[0]), float(pose[1]), float(pose[2])
        if self.start_pose is None:
            self.reset((x, y, heading))
            return self._status(reason="Start pose captured")

        if self.last_position is not None and not self.lap_complete:
            step = math.hypot(x - self.last_position[0], y - self.last_position[1])
            # Reject a single EKF discontinuity from falsely completing distance.
            if step <= config.MAX_EKF_LAP_STEP_METERS:
                self.distance_travelled += step
        self.last_position = (x, y)

        start_x, start_y, start_heading = self.start_pose
        distance_from_start = math.hypot(x - start_x, y - start_y)
        if distance_from_start >= config.LAP_DEPARTURE_RADIUS_METERS:
            self.departed_start = True

        just_completed = False
        if self.departed_start and not self.lap_complete:
            heading_error = abs(_normalize_angle(heading - start_heading))
            heading_limit = math.radians(
                config.MAXIMUM_LAP_RETURN_HEADING_ERROR_DEGREES
            )
            returned_to_start = distance_from_start <= config.LAP_START_RADIUS_METERS
            travelled_enough = (
                self.distance_travelled >= config.MINIMUM_LAP_DISTANCE_METERS
            )
            if returned_to_start and travelled_enough and heading_error <= heading_limit:
                self.lap_complete = True
                just_completed = True

        return self._status(just_completed=just_completed)

    def _status(self, just_completed: bool = False, reason: str = "") -> LapStatus:
        distance_from_start = 0.0
        if self.start_pose is not None and self.last_position is not None:
            distance_from_start = math.hypot(
                self.last_position[0] - self.start_pose[0],
                self.last_position[1] - self.start_pose[1],
            )
        return LapStatus(
            lap_complete=self.lap_complete,
            just_completed=just_completed,
            departed_start=self.departed_start,
            distance_travelled=self.distance_travelled,
            distance_from_start=distance_from_start,
            reason=reason,
        )

    @staticmethod
    def _pose_is_valid(pose: Sequence[float]) -> bool:
        return len(pose) >= 3 and all(math.isfinite(float(value)) for value in pose[:3])


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
