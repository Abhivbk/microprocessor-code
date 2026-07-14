import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import Delaunay, QhullError

from . import config


Point = Tuple[float, float]


@dataclass(frozen=True)
class Cone:
    x: float
    y: float
    cone_type: str
    hit_count: int = 0


@dataclass(frozen=True)
class TriangleResult:
    points: Tuple[Point, Point, Point]
    cone_types: Tuple[str, str, str]


@dataclass
class PlannerResult:
    triangles: List[TriangleResult] = field(default_factory=list)
    raw_centerline: List[Point] = field(default_factory=list)
    smoothed_centerline: List[Point] = field(default_factory=list)
    racing_line: List[Point] = field(default_factory=list)
    is_closed: bool = False
    is_valid: bool = False
    reason: str = ""

    @classmethod
    def invalid(cls, reason: str) -> "PlannerResult":
        return cls(is_valid=False, reason=reason)


class DelaunayPathPlanner:
    """Build an open first-lap centerline from confirmed EKF landmarks."""

    def __init__(
        self,
        track_width: float = config.TRACK_WIDTH_METERS,
        min_hit_count: int = config.MIN_LANDMARK_HIT_COUNT,
        planning_distance: float = config.PLANNING_DISTANCE_METERS,
        behind_distance: float = config.PLANNING_BEHIND_DISTANCE_METERS,
        lateral_limit: float = config.PLANNING_LATERAL_LIMIT_METERS,
        min_cross_width: float = config.MIN_CROSS_TRACK_WIDTH_METERS,
        max_cross_width: float = config.MAX_CROSS_TRACK_WIDTH_METERS,
        max_triangle_edge: float = config.MAX_TRIANGLE_EDGE_METERS,
        max_centerline_gap: float = config.MAX_CENTERLINE_GAP_METERS,
        smoothing_points: int = config.CENTERLINE_SMOOTHING_POINTS,
    ):
        self.track_width = track_width
        self.min_hit_count = min_hit_count
        self.planning_distance = planning_distance
        self.behind_distance = behind_distance
        self.lateral_limit = lateral_limit
        self.min_cross_width = min_cross_width
        self.max_cross_width = max_cross_width
        self.max_triangle_edge = max_triangle_edge
        self.max_centerline_gap = max_centerline_gap
        self.smoothing_points = smoothing_points

    def update(
        self,
        car_pose: Sequence[float],
        ekf_landmarks: Iterable[dict],
        completed_lap: bool = False,
    ) -> PlannerResult:
        if len(car_pose) < 3:
            return PlannerResult.invalid("Vehicle pose is unavailable")

        car_x, car_y, heading = map(float, car_pose[:3])
        cones = self._prepare_cones(
            ekf_landmarks,
            car_x,
            car_y,
            heading,
            use_global_map=completed_lap,
        )

        blue_count = sum(c.cone_type == "blue" for c in cones)
        yellow_count = sum(c.cone_type == "yellow" for c in cones)
        if len(cones) < config.MIN_TOTAL_BOUNDARY_CONES:
            return PlannerResult.invalid(
                f"At least {config.MIN_TOTAL_BOUNDARY_CONES} confirmed cones are required"
            )
        if (
            blue_count < config.MIN_CONES_PER_BOUNDARY_COLOR
            or yellow_count < config.MIN_CONES_PER_BOUNDARY_COLOR
        ):
            required = config.MIN_CONES_PER_BOUNDARY_COLOR
            return PlannerResult.invalid(
                f"At least {required} blue and {required} yellow cones are required"
            )

        points = np.array([(c.x, c.y) for c in cones], dtype=np.float64)
        try:
            simplices = Delaunay(points, qhull_options="QJ").simplices
        except (QhullError, ValueError) as exc:
            return PlannerResult.invalid(f"Delaunay triangulation failed: {exc}")

        valid_simplices = self._filter_triangles(simplices, cones)
        if not valid_simplices:
            return PlannerResult.invalid("No valid mixed-boundary triangles")

        cross_edges = self._find_cross_edges(valid_simplices, cones)
        if len(cross_edges) < config.MIN_VALID_CENTERLINE_POINTS:
            return PlannerResult.invalid("Not enough blue-yellow cross-track edges")

        triangle_results = [
            TriangleResult(
                points=tuple((cones[i].x, cones[i].y) for i in simplex),
                cone_types=tuple(cones[i].cone_type for i in simplex),
            )
            for simplex in valid_simplices
        ]

        edge_list = sorted(cross_edges)
        midpoints = np.array(
            [
                (
                    (cones[a].x + cones[b].x) / 2.0,
                    (cones[a].y + cones[b].y) / 2.0,
                )
                for a, b in edge_list
            ],
            dtype=np.float64,
        )
        adjacency = self._build_adjacency(valid_simplices, edge_list)
        ordered_indices = self._order_centerline(
            midpoints,
            adjacency,
            car_x,
            car_y,
            heading,
        )
        raw_centerline = [
            (float(midpoints[i, 0]), float(midpoints[i, 1]))
            for i in ordered_indices
        ]
        if len(raw_centerline) < config.MIN_VALID_CENTERLINE_POINTS:
            return PlannerResult(
                triangles=triangle_results,
                is_valid=False,
                reason="No connected centerline segment ahead of the vehicle",
            )

        if completed_lap:
            closure_gap = math.hypot(
                raw_centerline[-1][0] - raw_centerline[0][0],
                raw_centerline[-1][1] - raw_centerline[0][1],
            )
            if closure_gap > self.max_centerline_gap:
                return PlannerResult(
                    triangles=triangle_results,
                    raw_centerline=raw_centerline,
                    is_valid=False,
                    reason=f"Completed-lap centerline gap is {closure_gap:.2f} m",
                )
            closed_centerline = raw_centerline + [raw_centerline[0]]
            smoothed_centerline = self._smooth_closed_path(closed_centerline)
            racing_line, racing_reason = self._compute_racing_line(
                smoothed_centerline,
                cones,
            )
            return PlannerResult(
                triangles=triangle_results,
                raw_centerline=closed_centerline,
                smoothed_centerline=smoothed_centerline,
                racing_line=racing_line,
                is_closed=True,
                is_valid=(
                    len(smoothed_centerline) >= config.MIN_VALID_CENTERLINE_POINTS
                    and bool(racing_line)
                ),
                reason=racing_reason,
            )

        smoothed_centerline = self._smooth_open_path(raw_centerline)
        return PlannerResult(
            triangles=triangle_results,
            raw_centerline=raw_centerline,
            smoothed_centerline=smoothed_centerline,
            is_valid=len(smoothed_centerline) >= config.MIN_VALID_CENTERLINE_POINTS,
            reason="",
        )

    def _prepare_cones(
        self,
        landmarks: Iterable[dict],
        car_x: float,
        car_y: float,
        heading: float,
        use_global_map: bool = False,
    ) -> List[Cone]:
        prepared: List[Cone] = []
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for landmark in landmarks:
            cone_type = self._normalize_color(str(landmark.get("color", "")))
            hit_count = int(landmark.get("hit_count", 0))
            if cone_type is None or hit_count < self.min_hit_count:
                continue

            try:
                x = float(landmark["x"])
                y = float(landmark["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if not use_global_map:
                dx = x - car_x
                dy = y - car_y
                forward = dx * cos_h + dy * sin_h
                lateral = -dx * sin_h + dy * cos_h
                if not (-self.behind_distance <= forward <= self.planning_distance):
                    continue
                if abs(lateral) > self.lateral_limit:
                    continue

            candidate = Cone(x=x, y=y, cone_type=cone_type, hit_count=hit_count)
            if not self._is_duplicate(candidate, prepared):
                prepared.append(candidate)

        return prepared

    @staticmethod
    def _normalize_color(color: str) -> Optional[str]:
        normalized = color.lower()
        if "blue" in normalized:
            return "blue"
        if "yellow" in normalized:
            return "yellow"
        return None

    @staticmethod
    def _is_duplicate(
        candidate: Cone,
        cones: List[Cone],
        threshold: float = config.CONE_DUPLICATE_DISTANCE_METERS,
    ) -> bool:
        return any(
            candidate.cone_type == cone.cone_type
            and math.hypot(candidate.x - cone.x, candidate.y - cone.y) < threshold
            for cone in cones
        )

    def _filter_triangles(
        self,
        simplices: np.ndarray,
        cones: List[Cone],
    ) -> List[Tuple[int, int, int]]:
        valid: List[Tuple[int, int, int]] = []
        for simplex_values in simplices:
            simplex = tuple(int(i) for i in simplex_values)
            types = {cones[i].cone_type for i in simplex}
            if not {"blue", "yellow"}.issubset(types):
                continue

            edge_lengths = [
                math.hypot(
                    cones[a].x - cones[b].x,
                    cones[a].y - cones[b].y,
                )
                for a, b in self._triangle_edges(simplex)
            ]
            if max(edge_lengths) <= self.max_triangle_edge:
                valid.append(simplex)
        return valid

    def _find_cross_edges(
        self,
        simplices: List[Tuple[int, int, int]],
        cones: List[Cone],
    ) -> set:
        edges = set()
        for simplex in simplices:
            for a, b in self._triangle_edges(simplex):
                if cones[a].cone_type == cones[b].cone_type:
                    continue
                width = math.hypot(cones[a].x - cones[b].x, cones[a].y - cones[b].y)
                if self.min_cross_width <= width <= self.max_cross_width:
                    edges.add((min(a, b), max(a, b)))
        return edges

    @staticmethod
    def _triangle_edges(simplex: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
        a, b, c = simplex
        return ((a, b), (b, c), (c, a))

    def _build_adjacency(
        self,
        simplices: List[Tuple[int, int, int]],
        edge_list: List[Tuple[int, int]],
    ) -> dict:
        edge_indices = {edge: i for i, edge in enumerate(edge_list)}
        adjacency = {i: set() for i in range(len(edge_list))}

        for simplex in simplices:
            triangle_cross_edges = []
            for a, b in self._triangle_edges(simplex):
                edge = (min(a, b), max(a, b))
                if edge in edge_indices:
                    triangle_cross_edges.append(edge_indices[edge])

            for i in triangle_cross_edges:
                for j in triangle_cross_edges:
                    if i != j:
                        adjacency[i].add(j)
        return adjacency

    def _order_centerline(
        self,
        midpoints: np.ndarray,
        adjacency: dict,
        car_x: float,
        car_y: float,
        heading: float,
    ) -> List[int]:
        car = np.array([car_x, car_y], dtype=np.float64)
        heading_vector = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
        relative = midpoints - car
        forward = relative @ heading_vector
        start_candidates = np.where(
            forward >= config.CENTERLINE_START_MIN_FORWARD_METERS
        )[0]
        if not len(start_candidates):
            return []

        start = int(
            start_candidates[
                np.argmin(np.linalg.norm(relative[start_candidates], axis=1))
            ]
        )
        ordered = [start]
        visited = {start}
        travel_direction = heading_vector

        while True:
            current = ordered[-1]
            candidates = []
            for candidate in adjacency.get(current, set()):
                if candidate in visited:
                    continue
                delta = midpoints[candidate] - midpoints[current]
                distance = float(np.linalg.norm(delta))
                if distance < config.NUMERICAL_EPSILON or distance > self.max_centerline_gap:
                    continue
                direction = delta / distance
                alignment = float(direction @ travel_direction)
                if alignment < config.MIN_CENTERLINE_DIRECTION_ALIGNMENT:
                    continue
                candidates.append((alignment, -distance, candidate, direction))

            if not candidates:
                break

            _, _, next_index, next_direction = max(candidates, key=lambda item: (item[0], item[1]))
            ordered.append(next_index)
            visited.add(next_index)
            travel_direction = next_direction

        return ordered

    def _smooth_open_path(self, centerline: List[Point]) -> List[Point]:
        if len(centerline) < 4:
            return list(centerline)

        points = np.asarray(centerline, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.hstack(([True], segment_lengths > config.NUMERICAL_EPSILON))
        points = points[keep]
        if len(points) < 4:
            return [(float(x), float(y)) for x, y in points]

        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc_length = np.concatenate(([0.0], np.cumsum(distance)))
        sample_count = max(len(points), self.smoothing_points)
        samples = np.linspace(0.0, arc_length[-1], sample_count)

        spline_x = CubicSpline(arc_length, points[:, 0], bc_type="natural")
        spline_y = CubicSpline(arc_length, points[:, 1], bc_type="natural")
        return [
            (float(x), float(y))
            for x, y in zip(spline_x(samples), spline_y(samples))
        ]

    def _smooth_closed_path(self, centerline: List[Point]) -> List[Point]:
        if len(centerline) < 5:
            return list(centerline)

        points = np.asarray(centerline, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.hstack(([True], segment_lengths > config.NUMERICAL_EPSILON))
        points = points[keep]
        if len(points) < 5:
            return [(float(x), float(y)) for x, y in points]
        if not np.allclose(points[0], points[-1]):
            points = np.vstack((points, points[0]))

        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc_length = np.concatenate(([0.0], np.cumsum(distance)))
        spline_x = CubicSpline(arc_length, points[:, 0], bc_type="periodic")
        spline_y = CubicSpline(arc_length, points[:, 1], bc_type="periodic")
        samples = np.linspace(
            0.0,
            arc_length[-1],
            config.CLOSED_CENTERLINE_SMOOTHING_POINTS,
            endpoint=False,
        )
        smoothed = [
            (float(x), float(y))
            for x, y in zip(spline_x(samples), spline_y(samples))
        ]
        return smoothed + [smoothed[0]]

    def _compute_racing_line(
        self,
        smoothed_centerline: List[Point],
        cones: List[Cone],
    ) -> Tuple[List[Point], str]:
        if len(smoothed_centerline) < 5:
            return [], "Closed centerline is too short for racing-line optimization"

        points = np.asarray(smoothed_centerline[:-1], dtype=np.float64)
        normals = self._compute_normals(points)
        lower, upper = self._compute_boundary_offsets(points, normals, cones)
        q_matrix, linear = self._build_curvature_matrices(points, normals)
        q_matrix += np.eye(len(points)) * config.RACING_LINE_REGULARIZATION

        try:
            result = minimize(
                lambda alpha: float(alpha @ q_matrix @ alpha + 2.0 * linear @ alpha),
                np.zeros(len(points)),
                jac=lambda alpha: 2.0 * (q_matrix @ alpha + linear),
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={
                    "maxiter": config.RACING_LINE_OPTIMIZER_MAX_ITERATIONS,
                    "ftol": config.RACING_LINE_OPTIMIZER_FTOL,
                    "gtol": config.RACING_LINE_OPTIMIZER_GTOL,
                },
            )
        except Exception as exc:
            return [], f"Racing-line optimization failed: {exc}"

        if not result.success or not np.all(np.isfinite(result.x)):
            return [], f"Racing-line optimizer did not converge: {result.message}"

        racing_points = points + result.x[:, np.newaxis] * normals
        racing_line = [(float(x), float(y)) for x, y in racing_points]
        return racing_line + [racing_line[0]], ""

    @staticmethod
    def _compute_normals(points: np.ndarray) -> np.ndarray:
        tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        lengths = np.where(lengths < config.NUMERICAL_EPSILON, 1.0, lengths)
        tangents = tangents / lengths
        return np.column_stack((-tangents[:, 1], tangents[:, 0]))

    def _compute_boundary_offsets(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        cones: List[Cone],
    ) -> Tuple[np.ndarray, np.ndarray]:
        margin = config.RACING_LINE_SAFETY_MARGIN_METERS
        default_half_width = max(self.track_width / 2.0 - margin, 0.05)
        lower = np.full(len(points), -default_half_width)
        upper = np.full(len(points), default_half_width)

        blue = np.asarray([(c.x, c.y) for c in cones if c.cone_type == "blue"])
        yellow = np.asarray([(c.x, c.y) for c in cones if c.cone_type == "yellow"])
        if not len(blue) or not len(yellow):
            return lower, upper

        for index, point in enumerate(points):
            blue_point = blue[np.argmin(np.linalg.norm(blue - point, axis=1))]
            yellow_point = yellow[np.argmin(np.linalg.norm(yellow - point, axis=1))]
            blue_offset = float((blue_point - point) @ normals[index])
            yellow_offset = float((yellow_point - point) @ normals[index])
            bounded_low = min(blue_offset, yellow_offset) + margin
            bounded_high = max(blue_offset, yellow_offset) - margin
            if bounded_low <= bounded_high:
                lower[index] = bounded_low
                upper[index] = bounded_high
        return lower, upper

    @staticmethod
    def _build_curvature_matrices(
        points: np.ndarray,
        normals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        count = len(points)
        matrix = np.zeros((2 * count, count))
        base_second_difference = np.zeros(2 * count)

        for index in range(count):
            previous = (index - 1) % count
            following = (index + 1) % count
            for axis in range(2):
                row = 2 * index + axis
                matrix[row, previous] += normals[previous, axis]
                matrix[row, index] -= 2.0 * normals[index, axis]
                matrix[row, following] += normals[following, axis]
                base_second_difference[row] = (
                    points[following, axis]
                    - 2.0 * points[index, axis]
                    + points[previous, axis]
                )

        return matrix.T @ matrix, matrix.T @ base_second_difference
