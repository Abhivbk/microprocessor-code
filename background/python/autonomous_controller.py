import math
import time
import traceback
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

PATH_START_THROTTLE = 0.30
PATH_START_DURATION_S = 1.0
PATH_MOVING_SPEED_MPS = 0.20

class AutonomousController:
    def __init__(self, target_speed=4.0):
        # Kinematic Bicycle Model Constraints
        self.L = 1.53  # Wheelbase (m)
        self.max_steer_angle = math.radians(45.0)
        self.max_acceleration = 8.0  # m/s^2
        self.max_deceleration = 11.0  # m/s^2
        
        # Stanley Tuning Parameters
        self.k_e = 1.0  # Cross-track error gain (increase for tighter cornering)
        self.k_s = 1.0  # Softening constant (increase to prevent low-speed wobble)
        self.lookahead_dist = 1.5  # Heading lookahead (meters)
        
        self.target_speed = target_speed
        
        # State tracking
        self.last_target_point = None
        self.last_waypoints = []
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steering = 0.0
        self.progressive_path = False
        self.path_started_at = None
        self.path_evidence_count = 0
        self.last_status = "IDLE"
        self._last_error = None

    def _report_error(self, stage, error):
        """Print each distinct controller failure once instead of hiding it."""
        key = (stage, type(error).__name__, str(error))
        if key != self._last_error:
            self._last_error = key
            print(f"[Autonomy] {stage} failed: {error}", flush=True)
            traceback.print_exc()

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def extract_centerline(self, ekf_state, vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, max_track_width=20.0):
        """
        Uses Delaunay Triangulation to extract the centerline between blue and yellow cones.
        """
        self.path_evidence_count = 0
        blue_cones = [c for c in ekf_state if 'blue' in c['color']]
        yellow_cones = [c for c in ekf_state if 'yellow' in c['color']]
        all_cones = blue_cones + yellow_cones
        
        # Local Horizon Fix: Only triangulate cones in front of the car
        cones = []
        for c in all_cones:
            dx = c['x'] - vehicle_x
            dy = c['y'] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            if local_x > -1.0 and math.hypot(dx, dy) < 25.0:
                cones.append(c)
                
        waypoints = []
        
        if len(cones) >= 3:
            points = np.array([[c['x'], c['y']] for c in cones])
            colors = [c['color'] for c in cones]
            
            try:
                tri = Delaunay(points)
                cross_edges = set()
                
                for simplex in tri.simplices:
                    for i in range(3):
                        idx1 = simplex[i]
                        idx2 = simplex[(i + 1) % 3]
                        c1, c2 = colors[idx1], colors[idx2]
                        
                        if ('blue' in c1 and 'yellow' in c2) or ('yellow' in c1 and 'blue' in c2):
                            dist = self.get_distance(points[idx1], points[idx2])
                            if dist <= max_track_width:
                                cross_edges.add(tuple(sorted((idx1, idx2))))

                if not cross_edges:
                    for simplex in tri.simplices:
                        for i in range(3):
                            idx1 = simplex[i]
                            idx2 = simplex[(i + 1) % 3]
                            c1, c2 = colors[idx1], colors[idx2]
                            if ('blue' in c1 and 'yellow' in c2) or ('yellow' in c1 and 'blue' in c2):
                                dist = self.get_distance(points[idx1], points[idx2])
                                if dist <= max_track_width * 1.5:
                                    cross_edges.add(tuple(sorted((idx1, idx2))))
                                 
                for idx1, idx2 in cross_edges:
                    c1, c2 = colors[idx1], colors[idx2]
                    mx = (points[idx1][0] + points[idx2][0]) / 2.0
                    my = (points[idx1][1] + points[idx2][1]) / 2.0
            
                    if 'blue' in c1:
                        p_blue, p_yellow = points[idx1], points[idx2]
                    else:
                        p_blue, p_yellow = points[idx2], points[idx1]
                
                    v_dx = p_yellow[0] - p_blue[0]
                    v_dy = p_yellow[1] - p_blue[1]
            
                    fwd_dx = v_dy
                    fwd_dy = -v_dx
            
                    mag = math.hypot(fwd_dx, fwd_dy)
                    if mag > 0.001:
                        fwd_dx /= mag
                        fwd_dy /= mag
                        waypoints.append((mx, my, fwd_dx, fwd_dy))
            except Exception as error:
                self._report_error("Delaunay centreline", error)

        if not waypoints:
            return []

        # Directional Nearest Neighbor Sorting
        ordered_waypoints = []
        if waypoints:
            current_pt = min(waypoints, key=lambda p: math.hypot(p[0] - vehicle_x, p[1] - vehicle_y))
            ordered_waypoints.append(current_pt)
            unvisited = set(waypoints)
            unvisited.remove(current_pt)
            
            while unvisited:
                track_dx = current_pt[2]
                track_dy = current_pt[3]
                
                best_pt = None
                best_score = float('inf')
                
                for pt in unvisited:
                    vx = pt[0] - current_pt[0]
                    vy = pt[1] - current_pt[1]
                    dist = math.hypot(vx, vy)
                    
                    if dist > 0.001:
                        vx /= dist
                        vy /= dist
                    
                    dot = track_dx * vx + track_dy * vy
                    if dot < 0.0:
                        continue
                        
                    score = dist
                    if score < best_score:
                        best_score = score
                        best_pt = pt
                        
                if best_pt is None or best_score > 30.0:
                    break
                    
                ordered_waypoints.append(best_pt)
                unvisited.remove(best_pt)
                current_pt = best_pt

        self.path_evidence_count = len(ordered_waypoints)

        # B-Spline Smoothing
        if len(ordered_waypoints) >= 4:
            try:
                pts = np.array([(p[0], p[1]) for p in ordered_waypoints])
                _, idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(idx)]
                
                if len(pts) >= 4:
                    tck, u = splprep([pts[:, 0], pts[:, 1]], s=3.0, k=3)
                    u_new = np.linspace(0, 1, len(pts) * 3)
                    new_points = splev(u_new, tck)
                    ordered_waypoints = list(zip(new_points[0], new_points[1]))
                else:
                    ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
            except Exception as error:
                self._report_error("Centreline smoothing", error)
                ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
        else:
            ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]

        self.last_waypoints = ordered_waypoints
        return ordered_waypoints

    def extract_progressive_centerline(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta):
        """Build the best short path available before a full centreline exists."""
        orange = [c for c in ekf_state if "orange" in c["color"]]
        blue = [c for c in ekf_state if "blue" in c["color"]]
        yellow = [c for c in ekf_state if "yellow" in c["color"]]
        cos_h, sin_h = math.cos(vehicle_theta), math.sin(vehicle_theta)
        anchors, used_yellow = [], set()

        # Greedily pair each blue cone with its nearest sensible yellow cone.
        for left in blue:
            options = []
            for right in yellow:
                if id(right) in used_yellow:
                    continue
                width = self.get_distance((left["x"], left["y"]), (right["x"], right["y"]))
                mid = ((left["x"] + right["x"]) / 2.0, (left["y"] + right["y"]) / 2.0)
                local_x = (mid[0] - vehicle_x) * cos_h + (mid[1] - vehicle_y) * sin_h
                if 1.5 <= width <= 8.0 and 0.5 <= local_x <= 15.0:
                    options.append((width, local_x, mid, right))
            if options:
                _, local_x, midpoint, right = min(options, key=lambda item: item[0])
                used_yellow.add(id(right))
                anchors.append((local_x, midpoint))

        # Orange cones are an optional centre reference, never a requirement.
        visible_orange = []
        for cone in orange:
            local_x = (cone["x"] - vehicle_x) * cos_h + (cone["y"] - vehicle_y) * sin_h
            if -1.0 <= local_x <= 15.0:
                visible_orange.append((local_x, cone))
        if len(visible_orange) >= 2:
            group = sorted(visible_orange, key=lambda item: abs(item[0]))[:4]
            centre = (
                sum(item[1]["x"] for item in group) / len(group),
                sum(item[1]["y"] for item in group) / len(group),
            )
            centre_x = (centre[0] - vehicle_x) * cos_h + (centre[1] - vehicle_y) * sin_h
            anchors.append((centre_x, centre))

        if not anchors:
            self.path_evidence_count = 0
            self.last_status = "WAITING: no usable centreline waypoint"
            return []

        # Nearby observations represent the same path point; merge rather than stop.
        merged = []
        for _, point in sorted(anchors, key=lambda item: item[0]):
            if merged and self.get_distance(merged[-1], point) < 0.5:
                merged[-1] = ((merged[-1][0] + point[0]) / 2.0, (merged[-1][1] + point[1]) / 2.0)
            else:
                merged.append(point)

        waypoints = [(vehicle_x, vehicle_y)] + merged
        reference = waypoints[-2]
        dx, dy = waypoints[-1][0] - reference[0], waypoints[-1][1] - reference[1]
        distance = math.hypot(dx, dy)
        if distance < 0.2:
            dx, dy, distance = cos_h, sin_h, 1.0
        ux, uy = dx / distance, dy / distance
        extension = max(2.0, self.lookahead_dist)
        waypoints.append((waypoints[-1][0] + ux * extension, waypoints[-1][1] + uy * extension))
        self.path_evidence_count = len(merged)
        self.last_waypoints = waypoints
        return waypoints

    def mpc_control(self, current_speed, target_speed):
        """Optimize a short acceleration sequence for longitudinal speed."""
        horizon, dt = 5, 0.2
        initial = np.zeros(horizon)
        bounds = [(-self.max_deceleration, self.max_acceleration)] * horizon

        def cost_fn(accelerations):
            cost, speed = 0.0, current_speed
            for index, acceleration in enumerate(accelerations):
                speed += acceleration * dt
                cost += 10.0 * (speed - target_speed) ** 2 + acceleration ** 2
                if index > 0:
                    cost += 5.0 * (acceleration - accelerations[index - 1]) ** 2
            return cost

        try:
            result = minimize(cost_fn, initial, bounds=bounds, method="SLSQP")
            acceleration = result.x[0] if result.success else 1.5 * (target_speed - current_speed)
        except Exception as error:
            self._report_error("Longitudinal controller", error)
            acceleration = 1.5 * (target_speed - current_speed)

        acceleration = max(-self.max_deceleration, min(self.max_acceleration, acceleration))
        if acceleration > 0.0:
            return acceleration / self.max_acceleration, 0.0
        return 0.0, -acceleration / self.max_deceleration

    def stanley_control(self, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints):
        """Calculate lateral steering from heading and cross-track error."""
        if not waypoints:
            return self.last_steering
        closest_idx = target_local_y = None
        for index, point in enumerate(waypoints):
            dx, dy = point[0] - vehicle_x, point[1] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            local_y = -dx * math.sin(vehicle_theta) + dy * math.cos(vehicle_theta)
            if local_x > 0.0:
                closest_idx, target_local_y = index, local_y
                break
        if closest_idx is None:
            return self.last_steering

        target = waypoints[closest_idx]
        self.last_target_point = target
        lookahead_idx = closest_idx
        for index in range(closest_idx + 1, len(waypoints)):
            if self.get_distance(waypoints[index], target) >= self.lookahead_dist:
                lookahead_idx = index
                break
        if lookahead_idx > closest_idx:
            following = waypoints[lookahead_idx]
            path_yaw = math.atan2(following[1] - target[1], following[0] - target[0])
        else:
            path_yaw = vehicle_theta

        heading_error = (path_yaw - vehicle_theta + math.pi) % (2 * math.pi) - math.pi
        safe_speed = max(1.0, vehicle_speed)
        steering = -(heading_error + math.atan2(self.k_e * target_local_y, safe_speed + self.k_s))
        return max(-1.0, min(1.0, steering / 0.5))

    def compute_commands(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed):
        waypoints = self.extract_centerline(ekf_state, vehicle_x, vehicle_y, vehicle_theta)
        self.progressive_path = not waypoints
        if self.progressive_path:
            waypoints = self.extract_progressive_centerline(
                ekf_state, vehicle_x, vehicle_y, vehicle_theta,
            )
        if not waypoints:
            self.last_throttle = 0.0
            self.last_brake = 1.0
            self.path_started_at = None
            return self.last_throttle, self.last_steering, self.last_brake
        if self.path_started_at is None:
            self.path_started_at = time.monotonic()

        if self.path_evidence_count <= 1:
            target_speed = min(self.target_speed, 0.5)
        elif self.path_evidence_count < 4:
            target_speed = min(self.target_speed, 0.8)
        else:
            target_speed = self.target_speed

        closest_idx = 0
        for index, point in enumerate(waypoints):
            dx, dy = point[0] - vehicle_x, point[1] - vehicle_y
            if dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta) > 0.0:
                closest_idx = index
                break
        far_idx = closest_idx
        for index in range(closest_idx + 1, len(waypoints)):
            if self.get_distance(waypoints[index], waypoints[closest_idx]) >= 5.0:
                far_idx = index
                break
        if self.path_evidence_count >= 4 and far_idx > closest_idx and closest_idx + 1 < len(waypoints):
            far_yaw = math.atan2(
                waypoints[far_idx][1] - waypoints[closest_idx][1],
                waypoints[far_idx][0] - waypoints[closest_idx][0],
            )
            near_yaw = math.atan2(
                waypoints[closest_idx + 1][1] - waypoints[closest_idx][1],
                waypoints[closest_idx + 1][0] - waypoints[closest_idx][0],
            )
            yaw_difference = abs((far_yaw - near_yaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_difference > math.radians(20):
                target_speed = max(0.6, target_speed - 2.0 * (yaw_difference / math.radians(45)))

        throttle, brake = self.mpc_control(vehicle_speed, target_speed)
        if (vehicle_speed < PATH_MOVING_SPEED_MPS
                and time.monotonic() - self.path_started_at < PATH_START_DURATION_S):
            throttle, brake = max(throttle, PATH_START_THROTTLE), 0.0
        steering = self.stanley_control(vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints)

        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steering = steering
        if self.path_evidence_count <= 1:
            self.last_status = "SEED PATH: 1 observed waypoint"
        elif self.path_evidence_count < 4:
            self.last_status = f"POLYLINE: {self.path_evidence_count} observed waypoints"
        else:
            self.last_status = f"TRACKING: {self.path_evidence_count} centreline waypoints"
        return throttle, steering, brake
