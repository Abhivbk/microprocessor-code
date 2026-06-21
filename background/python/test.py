import json
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import os
import numpy as np
import cv2
import math
from ekf_slam import EKFSLAM

SHARED_MEM_DIR_FG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground"))
SHARED_MEM_DIR_BG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "background"))

CAM_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "cam.bin")
IMU_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "ekfin_imu_groundspeed_gyro.bin")
ACT_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "abs_current.bin")

CAM_CONES_PATH = os.path.join(SHARED_MEM_DIR_BG, "camera_cones.bin")
LIDAR_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "lid.bin")
CONTROLS_PATH = os.path.join(SHARED_MEM_DIR_BG, "control_instruction.bin")

CAM_W = 960
CAM_H = 540
LIDAR_W = 700
LIDAR_H = 540

class TestConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSDS Background Test Console")
        self.root.geometry("1550x950")
        self.root.configure(bg="#101010")

        self.latest_imu = None
        self.latest_actuator = None
        self.latest_cam_img = None
        self.latest_lidar_img = None
        
        self.latest_cam_cones = []
        self.latest_lidar_cones = []
        self.latest_fused_measurements = []

        self.imu_status = "Waiting for SHM"
        self.act_status = "Waiting for SHM"
        self.vision_status = "Waiting for SHM"
        self.ctrl_status = "Writing to SHM"

        self.desired = {
            "throttle": 0.0,
            "brake": 0.0,
            "steering": 0.0,
        }

        self.ekf = EKFSLAM()
        self.last_prediction_time = time.time()
        
        self.yolo_lock = threading.Lock()

        # Keyboard driving state
        self.pressed_keys = {}
        self.using_keyboard = False
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        self._build_ui()

        threading.Thread(target=self._shm_reader_thread, daemon=True).start()
        threading.Thread(target=self._shm_writer_thread, daemon=True).start()
        threading.Thread(target=self._fusion_thread, daemon=True).start()

        self.root.after(50, self.refresh_ui)

    def _build_ui(self):
        title = tk.Label(self.root, text="FSDS Background Test Console", font=("Arial", 20, "bold"), fg="cyan", bg="#101010")
        title.pack(pady=5)
        main = tk.Frame(self.root, bg="#101010")
        main.pack(fill="both", expand=True, padx=10, pady=5)
        left = tk.Frame(main, bg="#101010")
        left.pack(side="left", fill="both", expand=False)
        right = tk.Frame(main, bg="#101010")
        right.pack(side="right", fill="both", expand=True)
        col1 = tk.Frame(left, bg="#101010")
        col1.pack(side="left", fill="both", expand=False, padx=5)
        col2 = tk.Frame(left, bg="#101010")
        col2.pack(side="left", fill="both", expand=False, padx=5)

        self._build_controls_panel(col1)
        self._build_slam_panel(col1)
        self._build_status_panel(col2)
        self._build_imu_panel(col2)
        self._build_actuator_panel(col2)
        self._build_vision_panel(right)

    def _make_section(self, parent, title_text):
        frame = tk.LabelFrame(parent, text=title_text, fg="cyan", bg="#181818", font=("Arial", 12, "bold"), bd=2)
        frame.pack(fill="x", padx=8, pady=8)
        return frame

    def _build_slam_panel(self, parent):
        frame = self._make_section(parent, "EKF SLAM Measurements (Range, Bearing)")
        self.slam_text_var = tk.StringVar(value="No detections yet")
        tk.Label(frame, textvariable=self.slam_text_var, fg="yellow", bg="#181818", font=("Consolas", 10), justify="left", anchor="w").pack(fill="x", padx=12, pady=8)

    def _build_controls_panel(self, parent):
        frame = self._make_section(parent, "Control Output -> SHM")
        self.throttle_var = tk.DoubleVar(value=0.0)
        self.brake_var = tk.DoubleVar(value=0.0)
        self.steering_var = tk.DoubleVar(value=0.0)

        self.throttle_label = tk.StringVar(value="Throttle: 0.000")
        self.brake_label = tk.StringVar(value="Brake: 0.000")
        self.steering_label = tk.StringVar(value="Steering: 0.000")

        tk.Label(frame, textvariable=self.throttle_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(8, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.throttle_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)
        tk.Label(frame, textvariable=self.brake_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.brake_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)
        tk.Label(frame, textvariable=self.steering_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=-1.0, to=1.0, variable=self.steering_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)

        button_row = tk.Frame(frame, bg="#181818")
        button_row.pack(fill="x", padx=12, pady=12)
        tk.Button(button_row, text="Center Steering", command=self.center_steering, width=15).pack(side="left", padx=4)
        tk.Button(button_row, text="Zero Throttle", command=self.zero_throttle, width=15).pack(side="left", padx=4)
        tk.Button(button_row, text="Full Brake", command=self.full_brake, width=15).pack(side="left", padx=4)

        guide_text = "Keyboard Driving Controls (Focus this window):\n  - Throttle: W / Up Arrow\n  - Steering: A/D or Left/Right\n  - Brake: S / Down Arrow"
        tk.Label(frame, text=guide_text, fg="#aaaaaa", bg="#181818", font=("Arial", 9), justify="left", anchor="w").pack(fill="x", padx=12, pady=(0, 8))

    def _build_status_panel(self, parent):
        frame = self._make_section(parent, "SHM Status")
        self.imu_status_var = tk.StringVar(value="IMU: Disconnected")
        self.act_status_var = tk.StringVar(value="Actuator: Disconnected")
        self.vision_status_var = tk.StringVar(value="Vision: Disconnected")
        self.ctrl_status_var = tk.StringVar(value="Control TX: Writing")

        for var in [self.imu_status_var, self.act_status_var, self.vision_status_var, self.ctrl_status_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 11), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_imu_panel(self, parent):
        frame = self._make_section(parent, "SHM - IMU + Speed")
        self.speed_var = tk.StringVar(value="Ground Speed: ---")
        self.ang_var = tk.StringVar(value="Angular Vel: ---")
        self.lin_var = tk.StringVar(value="Linear Acc: ---")
        self.ori_var = tk.StringVar(value="Orientation: ---")

        for var in [self.speed_var, self.ang_var, self.lin_var, self.ori_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 10), justify="left", anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_actuator_panel(self, parent):
        frame = self._make_section(parent, "SHM - Actuator State")
        self.act_throttle_var = tk.StringVar(value="Throttle: ---")
        self.act_brake_var = tk.StringVar(value="Brake: ---")
        self.act_steering_var = tk.StringVar(value="Steering: ---")

        for var in [self.act_throttle_var, self.act_brake_var, self.act_steering_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 11), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_vision_panel(self, parent):
        frame = tk.LabelFrame(parent, text="Vision Stream", fg="cyan", bg="#181818", font=("Arial", 12, "bold"), bd=2)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.image_label = tk.Label(frame, bg="black")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

    def on_slider_change(self, _=None):
        self.desired["throttle"] = round(float(self.throttle_var.get()), 3)
        self.desired["brake"] = round(float(self.brake_var.get()), 3)
        self.desired["steering"] = round(float(self.steering_var.get()), 3)

    def center_steering(self):
        self.steering_var.set(0.0)
        self.on_slider_change()

    def zero_throttle(self):
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def full_brake(self):
        self.brake_var.set(1.0)
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def _on_key_press(self, event):
        self.pressed_keys[event.keysym] = True
        self.using_keyboard = True

    def _on_key_release(self, event):
        self.pressed_keys[event.keysym] = False

    def _update_keyboard_inputs(self):
        if not self.using_keyboard:
            return

        throttle = self.desired["throttle"]
        brake = self.desired["brake"]
        steering = self.desired["steering"]

        if self.pressed_keys.get("w") or self.pressed_keys.get("Up"):
            throttle = min(0.40, throttle + 0.05)
            brake = 0.0
        elif self.pressed_keys.get("s") or self.pressed_keys.get("Down"):
            throttle = 0.0
            brake = min(1.0, brake + 0.2)
        else:
            throttle = 0.0
            brake = 0.0

        if self.pressed_keys.get("space"):
            throttle = 0.0
            brake = 1.0

        if self.pressed_keys.get("a") or self.pressed_keys.get("Left"):
            steering = max(-1.0, steering - 0.1)
        elif self.pressed_keys.get("d") or self.pressed_keys.get("Right"):
            steering = min(1.0, steering + 0.1)
        else:
            if steering > 0:
                steering = max(0.0, steering - 0.15)
            elif steering < 0:
                steering = min(0.0, steering + 0.15)

        self.throttle_var.set(round(throttle, 3))
        self.brake_var.set(round(brake, 3))
        self.steering_var.set(round(steering, 3))
        self.on_slider_change()

    def _shm_reader_thread(self):
        while True:
            try:
                if os.path.exists(IMU_BIN_PATH):
                    import mmap
                    import struct
                    with open(IMU_BIN_PATH, "r+b") as f:
                        if os.fstat(f.fileno()).st_size >= 52:
                            ram = mmap.mmap(f.fileno(), 52, access=mmap.ACCESS_READ)
                            t, spd, ax, ay, az, lx, ly, lz, ox, oy, oz, ow = struct.unpack("<Q11f", ram[0:52])
                            self.latest_imu = {
                                "ground_speed_mps": spd,
                                "imu": {
                                    "angular_velocity": {"x": ax, "y": ay, "z": az},
                                    "linear_acceleration": {"x": lx, "y": ly, "z": lz},
                                    "orientation": {"x": ox, "y": oy, "z": oz, "w": ow}
                                }
                            }
                            self.imu_status = "Reading SHM (mmap)"
                            ram.close()
                    
                if os.path.exists(ACT_BIN_PATH):
                    import mmap
                    import struct
                    with open(ACT_BIN_PATH, "r+b") as f:
                        if os.fstat(f.fileno()).st_size >= 20:
                            ram = mmap.mmap(f.fileno(), 20, access=mmap.ACCESS_READ)
                            t, th, br, st = struct.unpack("<Qfff", ram[0:20])
                            self.latest_actuator = {"throttle": th, "brake": br, "steering": st}
                            self.act_status = "Reading SHM (mmap)"
                            ram.close()

                if os.path.exists(CAM_BIN_PATH):
                    raw = np.fromfile(CAM_BIN_PATH, dtype=np.uint8)
                    if len(raw) == CAM_W * CAM_H * 3:
                        cam_img = raw.reshape((CAM_H, CAM_W, 3))
                        self.vision_status = "Reading SHM"
                        with self.yolo_lock:
                            self.latest_cam_img = cam_img

                if os.path.exists(CAM_CONES_PATH):
                    import mmap
                    import struct
                    with open(CAM_CONES_PATH, "r+b") as f:
                        fs = os.fstat(f.fileno()).st_size
                        if fs >= 8:
                            ram = mmap.mmap(f.fileno(), fs, access=mmap.ACCESS_READ)
                            num_cones = struct.unpack("Q", ram[0:8])[0]
                            if fs >= 8 + num_cones * 24:
                                with self.yolo_lock:
                                    self.latest_cam_cones = []
                                    offset = 8
                                    labels = {0: "yellow_cone", 1: "blue_cone", 2: "orange_cone"}
                                    for _ in range(num_cones):
                                        x1, y1, x2, y2, conf, lid = struct.unpack("fffffi", ram[offset:offset+24])
                                        offset += 24
                                        bcx = (x1 + x2) / 2
                                        lbl = labels.get(lid, "unknown")
                                        self.latest_cam_cones.append((bcx, x1, y1, x2, y2, lbl, conf))
                            ram.close()

                if os.path.exists(LIDAR_BIN_PATH):
                    import mmap
                    import struct
                    with open(LIDAR_BIN_PATH, "r+b") as f:
                        file_size = os.fstat(f.fileno()).st_size
                        if file_size >= 8:
                            ram = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_READ)
                            num_points = struct.unpack("Q", ram[0:8])[0]
                            if file_size >= 8 + num_points * 16:
                                with self.yolo_lock:
                                    self.latest_lidar_cones = []
                                    offset = 8
                                    for _ in range(num_points):
                                        x = struct.unpack("d", ram[offset:offset+8])[0]
                                        offset += 8
                                        y = struct.unpack("d", ram[offset:offset+8])[0]
                                        offset += 8
                                        dist = math.sqrt(x*x + y*y)
                                        px = int(350 + y * 23.0)
                                        py = int(500 - x * 23.0)
                                        self.latest_lidar_cones.append((px, py, dist, y, x))
                            ram.close()

            except Exception as e:
                pass
            time.sleep(0.033)

    def _shm_writer_thread(self):
        os.makedirs(SHARED_MEM_DIR_BG, exist_ok=True)
        import struct
        import mmap
        
        BINARY_FORMAT = "<Qfff"
        STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

        if not os.path.exists(CONTROLS_PATH) or os.path.getsize(CONTROLS_PATH) != STRUCT_SIZE:
            with open(CONTROLS_PATH, "wb") as f:
                f.write(b"\x00" * STRUCT_SIZE)

        with open(CONTROLS_PATH, "r+b") as f:
            ram = mmap.mmap(f.fileno(), STRUCT_SIZE)
            while True:
                packed = struct.pack(BINARY_FORMAT, int(time.time()*1000), self.desired["throttle"], self.desired["brake"], self.desired["steering"])
                ram[0:STRUCT_SIZE] = packed
                self.ctrl_status = "Writing (mmap)"
                time.sleep(0.05)

    def _fusion_thread(self):
        while True:
            with self.yolo_lock:
                cam_cones = list(self.latest_cam_cones)
                lidar_cones = list(self.latest_lidar_cones)

            fused_cones = []
            used_lidar = set()

            for cam_cone in cam_cones:
                bcx, x1, y1, x2, y2, label, conf = cam_cone
                phi_cam = math.atan2(bcx - 480.0, 480.0)

                best_lidar_idx = -1
                for l_idx, lidar_cone in enumerate(lidar_cones):
                    if l_idx in used_lidar:
                        continue
                    lcx, lcy, dist, lat_m, fwd_m = lidar_cone
                    if fwd_m == 0:
                        continue
                    phi_lidar = math.atan2(lat_m, fwd_m)

                    angle_diff = abs(phi_cam - phi_lidar)
                    if angle_diff < 0.20:
                        best_lidar_idx = l_idx
                        break

                if best_lidar_idx != -1:
                    used_lidar.add(best_lidar_idx)
                    lcx, lcy, dist, lat_m, fwd_m = lidar_cones[best_lidar_idx]
                    fused_cones.append({
                        "label": label, "conf": conf, "cam_box": [int(x1), int(y1), int(x2), int(y2)],
                        "lidar_pixel": [int(lcx), int(lcy)], "range": dist, "bearing": -math.atan2(lat_m, fwd_m), "color": label
                    })
                else:
                    box_h = max(1.0, y2 - y1)
                    fallback_dist = (480.0 * 0.35) / box_h
                    if fallback_dist > 12.0:
                        continue
                    fused_cones.append({
                        "label": label, "conf": conf, "cam_box": [int(x1), int(y1), int(x2), int(y2)],
                        "lidar_pixel": None, "range": fallback_dist, "bearing": -phi_cam, "color": label
                    })

            with self.yolo_lock:
                self.latest_fused_measurements = fused_cones
            time.sleep(0.05)

    def refresh_ui(self):
        self._update_keyboard_inputs()

        now_time = time.time()
        dt = now_time - self.last_prediction_time
        self.last_prediction_time = now_time

        speed, yaw_rate, qz, qw = 0.0, 0.0, 0.0, 1.0
        if self.latest_imu:
            speed = self.latest_imu.get("ground_speed_mps", 0.0)
            yaw_rate = self.latest_imu.get("imu", {}).get("angular_velocity", {}).get("z", 0.0)
            ori = self.latest_imu.get("imu", {}).get("orientation", {})
            qz = ori.get("z", 0.0)
            qw = ori.get("w", 1.0)

        self.ekf.predict(speed, yaw_rate, dt)
        if qz != 0.0 or qw != 1.0:
            abs_theta = self.ekf.normalize_angle(2.0 * math.atan2(qz, qw))
            if not hasattr(self.ekf, 'heading_initialized'):
                self.ekf.x[2] = abs_theta
                self.ekf.heading_initialized = True
            else:
                self.ekf.update_heading(abs_theta)

        with self.yolo_lock:
            fused = list(self.latest_fused_measurements)

        if fused and speed > 0.1 and hasattr(self.ekf, 'heading_initialized'):
            self.ekf.update(fused)

        self.throttle_label.set(f"Throttle: {self.desired['throttle']:.3f}")
        self.brake_label.set(f"Brake: {self.desired['brake']:.3f}")
        self.steering_label.set(f"Steering: {self.desired['steering']:.3f}")

        self.imu_status_var.set(f"IMU: {self.imu_status}")
        self.act_status_var.set(f"Actuator: {self.act_status}")
        self.vision_status_var.set(f"Vision: {self.vision_status}")
        self.ctrl_status_var.set(f"Control TX: {self.ctrl_status}")

        if self.latest_imu:
            speed_val = self.latest_imu.get("ground_speed_mps", 0.0)
            imu = self.latest_imu.get("imu", {})
            av = imu.get("angular_velocity", {})
            la = imu.get("linear_acceleration", {})
            ori = imu.get("orientation", {})
            self.speed_var.set(f"Ground Speed: {speed_val:.3f} m/s")
            self.ang_var.set(f"Angular Vel: x={av.get('x', 0.0):+.4f}  y={av.get('y', 0.0):+.4f}  z={av.get('z', 0.0):+.4f}")
            self.lin_var.set(f"Linear Acc: x={la.get('x', 0.0):+.4f}  y={la.get('y', 0.0):+.4f}  z={la.get('z', 0.0):+.4f}")
            self.ori_var.set(f"Orientation: x={ori.get('x', 0.0):+.4f}  y={ori.get('y', 0.0):+.4f}  z={ori.get('z', 0.0):+.4f}  w={ori.get('w', 1.0):+.4f}")

        if self.latest_actuator:
            self.act_throttle_var.set(f"Throttle: {float(self.latest_actuator.get('throttle', 0.0)):.3f}")
            self.act_brake_var.set(f"Brake: {float(self.latest_actuator.get('brake', 0.0)):.3f}")
            self.act_steering_var.set(f"Steering: {float(self.latest_actuator.get('steering', 0.0)):.3f}")

        with self.yolo_lock:
            cam_img = self.latest_cam_img.copy() if self.latest_cam_img is not None else None
            l_cones = list(self.latest_lidar_cones)

        if cam_img is not None:
            # Recreate blank lidar img
            lidar_img = np.zeros((LIDAR_H, LIDAR_W, 3), dtype=np.uint8)
            center_x = LIDAR_W // 2
            center_y = LIDAR_H - 40
            scale = (LIDAR_H - 80) / 20.0
            
            # Draw Lidar lines
            for d in range(0, 21, 5):
                py = int(center_y - d * scale)
                cv2.line(lidar_img, (0, py), (LIDAR_W, py), (60,60,60), 1)
            for lateral in range(-20, 21, 5):
                px = int(center_x + lateral * scale)
                cv2.line(lidar_img, (px, 0), (px, LIDAR_H), (60,60,60), 1)
            cv2.circle(lidar_img, (center_x, center_y), 6, (0,255,255), -1)

            # Draw Lidar cones
            for c in l_cones:
                cx, cy, dist = c[0], c[1], c[2]
                cv2.circle(lidar_img, (cx, cy), 7, (0,165,255), 2)

            for cone in fused:
                label = cone["label"]
                conf = cone["conf"]
                cam_box = cone["cam_box"]
                lidar_pixel = cone["lidar_pixel"]

                if "yellow" in label.lower(): box_color = (255, 255, 0)
                elif "blue" in label.lower(): box_color = (0, 120, 255)
                elif "orange" in label.lower(): box_color = (255, 165, 0)
                else: box_color = (255, 255, 255)

                if lidar_pixel is not None:
                    cv2.circle(lidar_img, tuple(lidar_pixel), 10, (0, 255, 0), 2)
                    r_text = f" [{cone['range']:.1f}m]"
                else:
                    r_text = ""

                x1, y1, x2, y2 = cam_box
                cv2.rectangle(cam_img, (x1, y1), (x2, y2), box_color, 2)
                text = f"{label} {conf:.2f}{r_text}"
                cv2.putText(cam_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            ekf_img = self._draw_ekf_map()
            
            cam_img_rgb = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            lidar_img_rgb = cv2.cvtColor(lidar_img, cv2.COLOR_BGR2RGB)

            h_cam, w_cam, _ = cam_img_rgb.shape
            h_lid, w_lid, _ = lidar_img_rgb.shape
            h_ekf, w_ekf, _ = ekf_img.shape

            bottom_w = w_lid + w_ekf
            bottom_h = max(h_lid, h_ekf)
            bottom_row = np.zeros((bottom_h, bottom_w, 3), dtype=np.uint8)
            bottom_row[:h_lid, :w_lid] = lidar_img_rgb
            bottom_row[:h_ekf, w_lid:w_lid+w_ekf] = ekf_img

            dash_w = max(w_cam, bottom_w)
            dash_h = h_cam + bottom_h
            dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)

            dx_cam = (dash_w - w_cam) // 2
            dashboard[:h_cam, dx_cam:dx_cam+w_cam] = cam_img_rgb
            dx_bot = (dash_w - bottom_w) // 2
            dashboard[h_cam:, dx_bot:dx_bot+bottom_w] = bottom_row

            disp_img = Image.fromarray(dashboard)
            disp_img.thumbnail((1150, 850))
            tk_img = ImageTk.PhotoImage(disp_img)
            self.image_label.configure(image=tk_img)
            self.image_label.image = tk_img

        if fused:
            lines = []
            for cone in fused[:5]:
                r_val = f"{cone['range']:.2f}m" if cone['range'] is not None else "---"
                b_deg = math.degrees(cone['bearing'])
                lines.append(f"{cone['label'][:3].upper()}: r={r_val:<6} b={b_deg:+.1f}°")
            if len(fused) > 5: lines.append(f"... and {len(fused) - 5} more")
            self.slam_text_var.set("\n".join(lines))
        else:
            self.slam_text_var.set("No detections")

        self.root.after(50, self.refresh_ui)

    def _draw_ekf_map(self):
        map_img = np.zeros((540, 540, 3), dtype=np.uint8)
        map_img[:] = (16, 16, 16)
        cx, cy = 270, 270
        scale = 15.0
        xv, yv, theta = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        grid_spacing = 5.0
        start_x = (int(xv / grid_spacing) - 5) * grid_spacing
        end_x = (int(xv / grid_spacing) + 5) * grid_spacing
        start_y = (int(yv / grid_spacing) - 5) * grid_spacing
        end_y = (int(yv / grid_spacing) + 5) * grid_spacing
        for x_line in np.arange(start_x, end_x + grid_spacing, grid_spacing):
            px = int(cx + (yv - start_y) * scale)
            py = int(cy - (x_line - xv) * scale)
            cv2.line(map_img, (0, py), (540, py), (40, 40, 40), 1)
        for y_line in np.arange(start_y, end_y + grid_spacing, grid_spacing):
            px = int(cx + (y_line - yv) * scale)
            cv2.line(map_img, (px, 0), (px, 540), (40, 40, 40), 1)
        
        cv2.circle(map_img, (cx, cy), 6, (0, 255, 255), -1)
        dx = int(cx + 12 * math.sin(theta - self.ekf.x[2]))
        dy = int(cy - 12 * math.cos(theta - self.ekf.x[2]))
        cv2.line(map_img, (cx, cy), (dx, dy), (0, 255, 255), 2)

        for l_info in self.ekf.landmarks:
            idx = l_info["id"]
            color_str = l_info["color"].lower()
            lx, ly = self.ekf.x[3 + 2*idx], self.ekf.x[4 + 2*idx]
            if "yellow" in color_str: color = (255, 255, 0)
            elif "blue" in color_str: color = (0, 120, 255)
            elif "orange" in color_str: color = (255, 165, 0)
            else: color = (255, 255, 255)
            px = int(cx + (ly - yv) * scale)
            py = int(cy - (lx - xv) * scale)
            if 0 <= px < 540 and 0 <= py < 540:
                cv2.circle(map_img, (px, py), 4, color, -1)

        if len(self.ekf.trajectory) > 1:
            for i in range(len(self.ekf.trajectory) - 1):
                pt1 = self.ekf.trajectory[i]
                pt2 = self.ekf.trajectory[i+1]
                px1 = int(cx + (pt1[1] - yv) * scale)
                py1 = int(cy - (pt1[0] - xv) * scale)
                px2 = int(cx + (pt2[1] - yv) * scale)
                py2 = int(cy - (pt2[0] - xv) * scale)
                cv2.line(map_img, (px1, py1), (px2, py2), (255, 0, 255), 2)
        return map_img

if __name__ == "__main__":
    root = tk.Tk()
    app = TestConsoleApp(root)
    root.mainloop()