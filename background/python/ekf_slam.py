import numpy as np
import math


class EKFSLAM:
    def __init__(self, R_pose=None, Q_obs=None, init_p_pose=1e-3, init_p_landmark=5.0, assoc_threshold=5.0):
        """
        EKF SLAM core state estimation.
        - State vector x: [x_v, y_v, theta_v, x_1, y_1, ..., x_M, y_M]^T
        - Covariance matrix P
        """
        # Vehicle pose initialized at origin: [0, 0, 0]
        self.x = np.zeros(3, dtype=np.float64)

        # Covariance initialized to small values for vehicle pose
        self.P = np.diag([init_p_pose, init_p_pose, init_p_pose]).astype(np.float64)

        # Process noise covariance (for vehicle pose prediction step)
        # R_pose: noise in [x, y, theta] propagation
        if R_pose is None:
            self.R_pose = np.diag([0.02**2, 0.02**2, math.radians(1.5)**2]).astype(np.float64)
        else:
            self.R_pose = np.array(R_pose, dtype=np.float64)

        # Measurement noise covariance
        # Q_obs: noise in [range, bearing] observation
        if Q_obs is None:
            self.Q_obs = np.diag([0.15**2, math.radians(2.0)**2]).astype(np.float64)
        else:
            self.Q_obs = np.array(Q_obs, dtype=np.float64)

        self.init_p_landmark = init_p_landmark
        self.assoc_threshold = assoc_threshold

        # Landmark tracker to store association info and colors
        # list of dict: {"id": i, "color": color}
        self.landmarks = []

        # Keep track of history trace for trajectory rendering
        self.trajectory = []
        self.trajectory.append((float(self.x[0]), float(self.x[1])))

    def predict(self, v, omega, dt):
        """
        Predict vehicle pose using ground speed and yaw rate.
        v: ground speed (m/s)
        omega: angular velocity (rad/s)
        dt: time step (seconds)
        """
        if dt <= 0:
            return

        # Current state info
        xv, yv, theta = self.x[0], self.x[1], self.x[2]

        # 1. Update vehicle state
        self.x[0] = xv + v * math.cos(theta) * dt
        self.x[1] = yv + v * math.sin(theta) * dt
        self.x[2] = self.normalize_angle(theta + omega * dt)

        # 2. Update Covariance Matrix P
        # Jacobian G_t of motion model with respect to full state
        n_states = len(self.x)
        G = np.eye(n_states, dtype=np.float64)
        G[0, 2] = -v * math.sin(theta) * dt
        G[1, 2] = v * math.cos(theta) * dt

        # Predict covariance: P = G * P * G^T + R_t
        self.P = G @ self.P @ G.T
        self.P[0:3, 0:3] += self.R_pose

        # Store historical trace if car has moved significantly
        last_pt = self.trajectory[-1]
        if math.hypot(self.x[0] - last_pt[0], self.x[1] - last_pt[1]) > 0.05:
            self.trajectory.append((float(self.x[0]), float(self.x[1])))
            if len(self.trajectory) > 600:
                self.trajectory.pop(0)

    def update_heading(self, abs_theta):
        """
        Fuses absolute heading from IMU directly as a measurement, maintaining covariance correctly.
        """
        H = np.zeros((1, len(self.x)), dtype=np.float64)
        H[0, 2] = 1.0
        y_val = np.array([self.normalize_angle(abs_theta - self.x[2])], dtype=np.float64)
        R_heading = np.array([[math.radians(1.0)**2]], dtype=np.float64)
        
        S = H @ self.P @ H.T + R_heading
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_val
            self.P = (np.eye(len(self.x), dtype=np.float64) - K @ H) @ self.P
            self.x[2] = self.normalize_angle(self.x[2])
        except np.linalg.LinAlgError:
            pass

    def update(self, fused_measurements):
        """
        Update state using a list of fused measurements.
        fused_measurements: list of dicts: {"range": float, "bearing": float, "color": str}
        """
        for meas in fused_measurements:
            r = meas.get("range")
            b = meas.get("bearing")
            color = meas.get("color")
            if b is None or color is None:
                continue

            self._update_single(r, b, color)

    def _update_single(self, r, b, color):
        xv, yv, theta = self.x[0], self.x[1], self.x[2]

        n_landmarks = len(self.landmarks)

        # Data association: Find closest existing landmark of the SAME color class
        best_idx = -1
        best_dist = self.assoc_threshold

        for idx in range(n_landmarks):
            l_info = self.landmarks[idx]
            if l_info["color"] != color:
                continue

            lx_idx = 3 + 2 * idx
            ly_idx = 4 + 2 * idx
            lx, ly = self.x[lx_idx], self.x[ly_idx]

            # Predicted measurement for this landmark
            dx = lx - xv
            dy = ly - yv
            d2 = dx**2 + dy**2
            d = math.sqrt(d2)

            if d < 1e-5:
                continue

            pred_r = d
            pred_b = self.normalize_angle(math.atan2(dy, dx) - theta)

            # Measurement Jacobian H for this landmark
            H = np.zeros((2, len(self.x)), dtype=np.float64)
            # Derivative w.r.t vehicle pose
            H[0, 0] = -dx / d
            H[0, 1] = -dy / d
            H[0, 2] = 0.0
            H[1, 0] = dy / d2
            H[1, 1] = -dx / d2
            H[1, 2] = -1.0

            # Derivative w.r.t landmark coordinates
            H[0, lx_idx] = dx / d
            H[0, ly_idx] = dy / d
            H[1, lx_idx] = -dy / d2
            H[1, ly_idx] = dx / d2

            if r is not None:
                y_val = np.array([r - pred_r, self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H
                Q_i = self.Q_obs
            else:
                # Bearing only update
                y_val = np.array([self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H[1:2, :]
                Q_i = self.Q_obs[1:2, 1:2]

            # Innovation covariance: S = H * P * H^T + Q
            S = H_i @ self.P @ H_i.T + Q_i

            # Mahalanobis Distance
            try:
                S_inv = np.linalg.inv(S) if r is not None else np.array([[1.0 / S[0, 0]]])
                d_M = math.sqrt(y_val @ S_inv @ y_val)
            except np.linalg.LinAlgError:
                d_M = 999.0

            if d_M < best_dist:
                best_dist = d_M
                best_idx = idx

        if best_idx != -1:
            # Associate and update existing landmark
            idx = best_idx
            lx_idx = 3 + 2 * idx
            ly_idx = 4 + 2 * idx
            lx, ly = self.x[lx_idx], self.x[ly_idx]

            dx = lx - xv
            dy = ly - yv
            d2 = dx**2 + dy**2
            d = math.sqrt(d2)

            pred_r = d
            pred_b = self.normalize_angle(math.atan2(dy, dx) - theta)

            # Recompute Jacobian H
            H = np.zeros((2, len(self.x)), dtype=np.float64)
            H[0, 0] = -dx / d
            H[0, 1] = -dy / d
            H[0, 2] = 0.0
            H[1, 0] = dy / d2
            H[1, 1] = -dx / d2
            H[1, 2] = -1.0

            H[0, lx_idx] = dx / d
            H[0, ly_idx] = dy / d
            H[1, lx_idx] = -dy / d2
            H[1, ly_idx] = dx / d2

            if r is not None:
                y_val = np.array([r - pred_r, self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H
                Q_i = self.Q_obs
            else:
                y_val = np.array([self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H[1:2, :]
                Q_i = self.Q_obs[1:2, 1:2]

            S = H_i @ self.P @ H_i.T + Q_i
            try:
                S_inv = np.linalg.inv(S) if r is not None else np.array([[1.0 / S[0, 0]]])
                K = self.P @ H_i.T @ S_inv
                self.x = self.x + K @ y_val
                self.P = (np.eye(len(self.x), dtype=np.float64) - K @ H_i) @ self.P
                self.x[2] = self.normalize_angle(self.x[2])
            except np.linalg.LinAlgError:
                pass
                
            self.landmarks[idx]["hit_count"] += 1

        elif r is not None and len(self.landmarks) < 300:
            # Initialize new landmark ONLY if range is available and under cap
            # Global position of the observed cone (estimate)
            xl_est = xv + r * math.cos(theta + b)
            yl_est = yv + r * math.sin(theta + b)
            
            self.x = np.append(self.x, [xl_est, yl_est])

            # Expand Covariance Matrix
            old_size = len(self.P)
            new_P = np.zeros((old_size + 2, old_size + 2), dtype=np.float64)
            new_P[0:old_size, 0:old_size] = self.P

            # Add landmark covariance block
            new_P[old_size, old_size] = self.init_p_landmark
            new_P[old_size + 1, old_size + 1] = self.init_p_landmark

            self.P = new_P

            # Add landmark metadata
            self.landmarks.append({
                "id": n_landmarks,
                "color": color,
                "hit_count": 1
            })

    @staticmethod
    def normalize_angle(angle):
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
