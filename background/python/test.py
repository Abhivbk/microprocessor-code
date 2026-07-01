import struct
import json
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import os
import numpy as np
from ultralytics import YOLO
import cv2
import math
import queue
import mmap
from ekf_slam import EKFSLAM
from autonomous_controller import AutonomousController


SHARED_MEM_DIR_FG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground"))
SHARED_MEM_DIR_BG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "background"))

CAM_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "cam.bin")
IMU_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "ekfin_imu_groundspeed_gyro.bin")
ACT_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "abs_current.bin")
LIDAR_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "lid.bin")
CONTROLS_PATH = os.path.join(SHARED_MEM_DIR_BG, "control_instruction.bin")


class TestConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSDS Test Console")
        self.root.geometry("1550x950")
        self.root.configure(bg="#101010")

        self.latest_imu = None
        self.imu_feedback_fresh = False
        self.latest_actuator = None
        self.latest_image = None

        _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        try:
            self.yolo_model = YOLO(_model_path)
            print(f"[YOLO] model loaded from {_model_path}")
        except Exception as _e:
            self.yolo_model = None
            print(f"[YOLO] failed to load model: {_e}")

        self.imu_status = "Disconnected"
        self.act_status = "Disconnected"
        self.vision_status = "Disconnected"
        self.ctrl_status = "Disconnected"

        self.manual_desired = {
            "throttle": 0.0,
            "brake": 1.0,
            "steering": 0.0,
        }

        # Initialize EKF SLAM state estimator
        self.ekf = EKFSLAM()
        self.last_prediction_time = time.time()
        self.new_fused_ready = False

        # Initialize Autonomous Controller (Speed capped to 1.0 m/s)
        self.controller = AutonomousController(target_speed=1.0)
        self.autonomous_mode = False

        # Dynamic Tuning variables initialized from self.controller defaults
        self.tune_L_var = tk.DoubleVar(value=self.controller.L)
        self.tune_max_steer_var = tk.DoubleVar(value=math.degrees(self.controller.max_steer_angle))
        self.tune_max_accel_var = tk.DoubleVar(value=self.controller.max_acceleration)
        self.tune_max_decel_var = tk.DoubleVar(value=self.controller.max_deceleration)
        self.tune_k_e_var = tk.DoubleVar(value=self.controller.k_e)
        self.tune_k_s_var = tk.DoubleVar(value=self.controller.k_s)
        self.tune_lookahead_var = tk.DoubleVar(value=self.controller.lookahead_dist)
        self.tune_target_speed_var = tk.DoubleVar(value=self.controller.target_speed)

        # Thread-safe image processing pipeline
        self.yolo_input_queue = queue.Queue(maxsize=1)
        self.yolo_lock = threading.Lock()
        self.latest_cam_cones = []
        self.latest_fused_measurements = []
        self.latest_cam_img = None
        self.latest_lidar_img = None

        # Keyboard driving state
        self.pressed_keys = {}
        self.using_keyboard = False
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        self._build_ui()
        self._create_map_window()

        threading.Thread(target=self._shm_reader_thread, daemon=True).start()
        threading.Thread(target=self._shm_writer_thread, daemon=True).start()
        threading.Thread(target=self._yolo_worker_thread, daemon=True).start()

        self.root.after(50, self.refresh_ui)

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="FSDS SHM Test Console",
            font=("Arial", 20, "bold"),
            fg="cyan",
            bg="#101010"
        )
        title.pack(pady=5)

        main = tk.Frame(self.root, bg="#101010")
        main.pack(fill="both", expand=True, padx=10, pady=5)

        left = tk.Frame(main, bg="#101010")
        left.pack(side="left", fill="both", expand=False)
        right = tk.Frame(main, bg="#101010")
        right.pack(side="right", fill="both", expand=True)

        # Split left panel into three columns for layout compacting on laptop screens
        col1 = tk.Frame(left, bg="#101010")
        col1.pack(side="left", fill="both", expand=False, padx=5)

        col2 = tk.Frame(left, bg="#101010")
        col2.pack(side="left", fill="both", expand=False, padx=5)

        col3 = tk.Frame(left, bg="#101010")
        col3.pack(side="left", fill="both", expand=False, padx=5)

        self._build_controls_panel(col1)
        self._build_slam_panel(col1)

        self._build_status_panel(col2)
        self._build_imu_panel(col2)
        self._build_actuator_panel(col2)

        self._build_tuning_panel(col3)

        self._build_vision_panel(right)

    def _build_slam_panel(self, parent):
        frame = self._make_section(parent, "EKF SLAM Measurements (Range, Bearing)")
        self.slam_text_var = tk.StringVar(value="No detections yet")
        tk.Label(
            frame,
            textvariable=self.slam_text_var,
            fg="yellow",
            bg="#181818",
            font=("Consolas", 10),
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(8, 4))

        tk.Button(
            frame,
            text="Show EKF SLAM Map Window",
            command=self.show_map_window,
            bg="#2c2c2c",
            fg="white",
            activebackground="cyan"
        ).pack(pady=(0, 8))

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
        self.brake_var = tk.DoubleVar(value=1.0)
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

        # Keyboard drive guide
        guide_text = (
            "Keyboard Driving Controls (Focus this window):\n"
            "  - Throttle: W / Up Arrow (limit: 0.40)\n"
            "  - Steering: A/D or Left/Right Arrow\n"
            "  - Brake: S / Down Arrow\n"
            "  - E-Brake: Space\n"
            "  - Toggle Auto Mode: P"
        )
        tk.Label(
            frame,
            text=guide_text,
            fg="#aaaaaa",
            bg="#181818",
            font=("Arial", 9),
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 8))

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
        frame = tk.LabelFrame(
            parent,
            text="SHM - Vision Stream",
            fg="cyan",
            bg="#181818",
            font=("Arial", 12, "bold"),
            bd=2
        )
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.image_label = tk.Label(frame, bg="black")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_tuning_panel(self, parent):
        frame = self._make_section(parent, "Autonomous Controller Tuning")

        # Helper to add a slider
        def add_tuning_slider(label_var, double_var, from_val, to_val):
            tk.Label(frame, textvariable=label_var, fg="white", bg="#181818", font=("Arial", 10)).pack(anchor="w", padx=12, pady=(4, 1))
            scale = ttk.Scale(frame, from_=from_val, to=to_val, variable=double_var, orient="horizontal")
            scale.pack(fill="x", padx=12, pady=(0, 4))
            return scale

        self.lbl_L = tk.StringVar()
        self.lbl_max_steer = tk.StringVar()
        self.lbl_max_accel = tk.StringVar()
        self.lbl_max_decel = tk.StringVar()
        self.lbl_k_e = tk.StringVar()
        self.lbl_k_s = tk.StringVar()
        self.lbl_lookahead = tk.StringVar()
        self.lbl_target_speed = tk.StringVar()

        # Add sliders
        add_tuning_slider(self.lbl_L, self.tune_L_var, 1.0, 3.0)
        add_tuning_slider(self.lbl_max_steer, self.tune_max_steer_var, 10.0, 90.0)
        add_tuning_slider(self.lbl_max_accel, self.tune_max_accel_var, 1.0, 20.0)
        add_tuning_slider(self.lbl_max_decel, self.tune_max_decel_var, 1.0, 20.0)
        add_tuning_slider(self.lbl_k_e, self.tune_k_e_var, 0.1, 10.0)
        add_tuning_slider(self.lbl_k_s, self.tune_k_s_var, 0.1, 10.0)
        add_tuning_slider(self.lbl_lookahead, self.tune_lookahead_var, 0.5, 10.0)
        add_tuning_slider(self.lbl_target_speed, self.tune_target_speed_var, 0.5, 10.0)

        # Set trace on all DoubleVars so they update self.controller immediately on any change!
        for var in [self.tune_L_var, self.tune_max_steer_var, self.tune_max_accel_var, self.tune_max_decel_var,
                    self.tune_k_e_var, self.tune_k_s_var, self.tune_lookahead_var, self.tune_target_speed_var]:
            var.trace_add("write", self.on_tune_change)

        # Initialize labels and controller
        self.on_tune_change()

        # Reset button
        btn_reset = tk.Button(frame, text="Reset to Defaults", command=self.reset_tune_defaults, width=20, bg="#2c2c2c", fg="white", activebackground="cyan")
        btn_reset.pack(pady=10)

    def on_tune_change(self, *args):
        if not hasattr(self, 'controller'):
            return
        try:
            L = float(self.tune_L_var.get())
            max_steer = float(self.tune_max_steer_var.get())
            max_accel = float(self.tune_max_accel_var.get())
            max_decel = float(self.tune_max_decel_var.get())
            k_e = float(self.tune_k_e_var.get())
            k_s = float(self.tune_k_s_var.get())
            lookahead = float(self.tune_lookahead_var.get())
            target_speed = float(self.tune_target_speed_var.get())
        except (tk.TclError, ValueError):
            return

        self.controller.L = L
        self.controller.max_steer_angle = math.radians(max_steer)
        self.controller.max_acceleration = max_accel
        self.controller.max_deceleration = max_decel
        self.controller.k_e = k_e
        self.controller.k_s = k_s
        self.controller.lookahead_dist = lookahead
        self.controller.target_speed = target_speed

        # Also update label texts dynamically
        self.lbl_L.set(f"Wheelbase (L): {L:.2f} m")
        self.lbl_max_steer.set(f"Max Steer Angle: {max_steer:.1f}°")
        self.lbl_max_accel.set(f"Max Accel: {max_accel:.1f} m/s²")
        self.lbl_max_decel.set(f"Max Decel: {max_decel:.1f} m/s²")
        self.lbl_k_e.set(f"Stanley k_e: {k_e:.2f}")
        self.lbl_k_s.set(f"Stanley k_s: {k_s:.2f}")
        self.lbl_lookahead.set(f"Lookahead Dist: {lookahead:.2f} m")
        self.lbl_target_speed.set(f"Target Speed: {target_speed:.2f} m/s")

    def reset_tune_defaults(self):
        defaults = AutonomousController(target_speed=1.0)
        self.tune_L_var.set(defaults.L)
        self.tune_max_steer_var.set(math.degrees(defaults.max_steer_angle))
        self.tune_max_accel_var.set(defaults.max_acceleration)
        self.tune_max_decel_var.set(defaults.max_deceleration)
        self.tune_k_e_var.set(defaults.k_e)
        self.tune_k_s_var.set(defaults.k_s)
        self.tune_lookahead_var.set(defaults.lookahead_dist)
        self.tune_target_speed_var.set(defaults.target_speed)
        self.on_tune_change()

    def _create_map_window(self):
        self.map_window = tk.Toplevel(self.root)
        self.map_window.title("EKF SLAM Map")
        self.map_window.geometry("560x560")
        self.map_window.configure(bg="#101010")
        
        self.map_label = tk.Label(self.map_window, bg="black")
        self.map_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Intercept close window event to hide it instead of destroying it
        self.map_window.protocol("WM_DELETE_WINDOW", self._on_map_window_close)

    def show_map_window(self):
        self.map_window.deiconify()
        self.map_window.lift()

    def _on_map_window_close(self):
        self.map_window.withdraw()

    def on_slider_change(self, _=None):
        self.manual_desired["throttle"] = round(float(self.throttle_var.get()), 3)
        self.manual_desired["brake"] = round(float(self.brake_var.get()), 3)
        self.manual_desired["steering"] = round(float(self.steering_var.get()), 3)
        if not self.autonomy_enabled:
            self._set_desired(self.manual_desired)

    def center_steering(self):
        self.steering_var.set(0.0)
        self.on_slider_change()

    def zero_throttle(self):
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def full_brake(self):
        if self.autonomy_enabled:
            self._disable_autonomy("Manual emergency brake")
        self.brake_var.set(1.0)
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def _shm_reader_thread(self):
        while True:
            try:
                if os.path.exists(IMU_BIN_PATH):
                    try:
                        with open(IMU_BIN_PATH, "rb") as f:
                            if os.fstat(f.fileno()).st_size >= 52:
                                ram = mmap.mmap(f.fileno(), 52, access=mmap.ACCESS_READ)
                                t, spd, ax, ay, az, lx, ly, lz, ox, oy, oz, ow = struct.unpack("<Q11f", ram[0:52])
                                self.latest_imu = {
                                    "timestamp_ms": t,
                                    "ground_speed_mps": spd,
                                    "imu": {
                                        "angular_velocity": {"x": ax, "y": ay, "z": az},
                                        "linear_acceleration": {"x": lx, "y": ly, "z": lz},
                                        "orientation": {"x": ox, "y": oy, "z": oz, "w": ow}
                                    }
                                }
                                self.imu_status = "Reading SHM (mmap)"
                                ram.close()
                    except Exception:
                        pass
                    
                if os.path.exists(ACT_BIN_PATH):
                    try:
                        with open(ACT_BIN_PATH, "rb") as f:
                            if os.fstat(f.fileno()).st_size >= 20:
                                ram = mmap.mmap(f.fileno(), 20, access=mmap.ACCESS_READ)
                                t, th, br, st = struct.unpack("<Qfff", ram[0:20])
                                self.latest_actuator = {
                                    "timestamp_ms": t,
                                    "throttle": th,
                                    "brake": br,
                                    "steering": st
                                }
                                self.act_status = "Fresh feedback (mmap)"
                                ram.close()
                    except Exception:
                        pass

                if os.path.exists(CAM_BIN_PATH):
                    try:
                        with open(CAM_BIN_PATH, "rb") as f:
                            file_size = os.fstat(f.fileno()).st_size
                            if file_size > 32:
                                ram = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_READ)
                                height, width, channels, dtype_code = struct.unpack("QQQQ", ram[0:32])
                                img_size = int(height) * int(width) * int(channels)
                                if file_size >= 32 + img_size and img_size > 0:
                                    raw_bytes = bytes(ram[32:32 + img_size])
                                    cam_img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((int(height), int(width), int(channels)))
                                    self.vision_status = "Reading SHM (mmap)"
                                    
                                    # Process LiDAR here to keep sync
                                    lidar_cones = []
                                    if os.path.exists(LIDAR_BIN_PATH):
                                        try:
                                            with open(LIDAR_BIN_PATH, "rb") as lf:
                                                lf_size = os.fstat(lf.fileno()).st_size
                                                if lf_size >= 8:
                                                    lram = mmap.mmap(lf.fileno(), lf_size, access=mmap.ACCESS_READ)
                                                    num_points = struct.unpack("Q", lram[0:8])[0]
                                                    if lf_size >= 8 + num_points * 16:
                                                        offset = 8
                                                        LIDAR_W = 700
                                                        LIDAR_H = 540
                                                        LIDAR_RANGE_METERS = 20.0
                                                        center_x = LIDAR_W // 2
                                                        center_y = LIDAR_H - 40
                                                        scale = (LIDAR_H - 80) / LIDAR_RANGE_METERS
                                                        for _ in range(num_points):
                                                            x = struct.unpack("d", lram[offset:offset+8])[0]
                                                            offset += 8
                                                            y = struct.unpack("d", lram[offset:offset+8])[0]
                                                            offset += 8
                                                            dist = math.sqrt(x*x + y*y)
                                                            px = int(center_x + y * scale)
                                                            py = int(center_y - x * scale)
                                                            lidar_cones.append((px, py, dist, y, x))
                                                    lram.close()
                                        except Exception:
                                            pass

                                    with self.yolo_lock:
                                        self.latest_cam_img = cam_img.copy()
                                        self.latest_lidar_img = np.zeros((LIDAR_H, LIDAR_W, 3), dtype=np.uint8) # mock lidar_img for safety if needed
                                        
                                    try:
                                        self.yolo_input_queue.put_nowait((cam_img, lidar_cones))
                                    except queue.Full:
                                        try:
                                            self.yolo_input_queue.get_nowait()
                                            self.yolo_input_queue.put_nowait((cam_img, lidar_cones))
                                        except Exception:
                                            pass
                                            
                                ram.close()
                    except Exception:
                        pass
            except Exception as e:
                pass
            time.sleep(0.033)

    def _shm_writer_thread(self):
        os.makedirs(SHARED_MEM_DIR_BG, exist_ok=True)
        BINARY_FORMAT = "<Qfff"
        STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

        if not os.path.exists(CONTROLS_PATH) or os.path.getsize(CONTROLS_PATH) != STRUCT_SIZE:
            with open(CONTROLS_PATH, "wb") as f:
                f.write(b"\x00" * STRUCT_SIZE)

        with open(CONTROLS_PATH, "r+b") as f:
            ram = mmap.mmap(f.fileno(), STRUCT_SIZE)
            while True:
                if self.autonomous_mode and not self.using_keyboard:
                    ekf_state = []
                    for idx, l_info in enumerate(self.ekf.landmarks):
                        if l_info.get("hit_count", 0) >= 3:
                            lx_idx = 3 + 2 * idx
                            ly_idx = 4 + 2 * idx
                            if lx_idx < len(self.ekf.x):
                                ekf_state.append({
                                    "x": float(self.ekf.x[lx_idx]),
                                    "y": float(self.ekf.x[ly_idx]),
                                    "color": l_info["color"].lower()
                                })
                    
                    speed = self.latest_imu.get("ground_speed_mps", 0.0) if self.latest_imu else 0.0
                    
                    thr, strng, brk = self.controller.compute_commands(
                        ekf_state, self.ekf.x[0], self.ekf.x[1], self.ekf.x[2], speed
                    )
                    self.desired['throttle'] = thr
                    self.desired['steering'] = strng
                    self.desired['brake'] = brk
                
                command = dict(self.desired)
                command_timestamp_ms = int(time.time() * 1000)
                packed = struct.pack(
                    BINARY_FORMAT,
                    command_timestamp_ms,
                    command["throttle"],
                    command["brake"],
                    command["steering"],
                )
                ram[0:STRUCT_SIZE] = packed
                self.ctrl_status = "Writing (mmap)"
                time.sleep(0.05)

    def _yolo_worker_thread(self):
        while True:
            try:
                cam_img, lidar_cones = self.yolo_input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self.yolo_model is not None:
                try:
                    # This YOLOv8 model expects RGB numpy arrays (due to color-swapped training dataset)
                    results = self.yolo_model(cam_img, verbose=False)[0]

                    # Extract YOLO Camera cones
                    raw_cam_cones = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = results.names.get(cls_id, str(cls_id))
                        
                        # 1. Low-Confidence Detections Filter
                        if conf < 0.60:
                            continue
                            
                        # 2. Edge-of-Frame Truncation (Bearing Skew)
                        CAM_W = 960
                        # Increased margin to 5px since YOLO boxes don't always perfectly touch the 959th pixel
                        if x1 <= 5 or x2 >= CAM_W - 5:
                            continue
                            
                        bcx = (x1 + x2) / 2
                        raw_cam_cones.append((bcx, x1, y1, x2, y2, label, conf))

                    # 3. Cross-Class Bounding Box Overlap (NMS)
                    cam_cones = []
                    # Sort by confidence descending
                    raw_cam_cones.sort(key=lambda c: c[6], reverse=True)
                    for new_cone in raw_cam_cones:
                        _, x1_n, y1_n, x2_n, y2_n, _, _ = new_cone
                        overlap = False
                        for kept_cone in cam_cones:
                            _, x1_k, y1_k, x2_k, y2_k, _, _ = kept_cone
                            # IoU calculation
                            xx1 = max(x1_n, x1_k)
                            yy1 = max(y1_n, y1_k)
                            xx2 = min(x2_n, x2_k)
                            yy2 = min(y2_n, y2_k)
                            w = max(0, xx2 - xx1)
                            h = max(0, yy2 - yy1)
                            inter = w * h
                            area_n = (x2_n - x1_n) * (y2_n - y1_n)
                            area_k = (x2_k - x1_k) * (y2_k - y1_k)
                            iou = inter / float(area_n + area_k - inter)
                            # Decreased IoU threshold to 0.3 to aggressively suppress overlapping 
                            # boxes of different classes that might have slightly different shapes
                            if iou > 0.3:  # Overlap threshold
                                overlap = True
                                break
                        if not overlap:
                            cam_cones.append(new_cone)

                    # Sort cam_cones from bottom-to-top (closest to farthest in 2D image)
                    cam_cones.sort(key=lambda c: c[4], reverse=True)
                    # Sort lidar_cones from closest to farthest (3D distance)
                    lidar_cones.sort(key=lambda l: l[2])

                    fused_cones = []
                    used_lidar = set()

                    for cam_cone in cam_cones:
                        bcx, x1, y1, x2, y2, label, conf = cam_cone
                        # 90 degrees FOV camera: u = bcx, u_c = 480, f = 480
                        phi_cam = math.atan2(bcx - 480.0, 480.0)

                        best_lidar_idx = -1

                        for l_idx, lidar_cone in enumerate(lidar_cones):
                            if l_idx in used_lidar:
                                continue
                            lcx, lcy, dist, lat_m, fwd_m = lidar_cone
                            phi_lidar = math.atan2(lat_m, fwd_m)

                            angle_diff = abs(phi_cam - phi_lidar)
                            if angle_diff < 0.20:  # Matches within ~11.5 degrees
                                best_lidar_idx = l_idx
                                break  # Take the first (closest) available LiDAR cone!

                        if best_lidar_idx != -1:
                            used_lidar.add(best_lidar_idx)
                            lcx, lcy, dist, lat_m, fwd_m = lidar_cones[best_lidar_idx]
                            fused_cones.append({
                                "label": label,
                                "conf": conf,
                                "cam_box": [int(x1), int(y1), int(x2), int(y2)],
                                "lidar_pixel": [int(lcx), int(lcy)],
                                "range": dist,
                                "bearing": -math.atan2(lat_m, fwd_m),  # CCW positive
                                "color": label
                            })
                        else:
                            # Fallback: Monocular Depth Estimation
                            # Z = (f * H) / h, where f=480, H=0.35m (approx cone height), h = y2 - y1
                            box_h = max(1.0, y2 - y1)
                            fallback_dist = (480.0 * 0.35) / box_h
                            
                            # 4. Sensor Fusion Range Mismatch (Horizon)
                            # Do not map distant monocular estimations without LiDAR confirmation
                            if fallback_dist > 12.0:
                                continue
                                
                            fused_cones.append({
                                "label": label,
                                "conf": conf,
                                "cam_box": [int(x1), int(y1), int(x2), int(y2)],
                                "lidar_pixel": None,
                                "range": fallback_dist,
                                "bearing": -phi_cam,  # CCW positive
                                "color": label
                            })

                    with self.yolo_lock:
                        self.latest_cam_cones = cam_cones
                        self.latest_fused_measurements = fused_cones
                        self.new_fused_ready = True

                except Exception as e:
                    print(f"[YOLO Thread] Error: {e}")

            self.yolo_input_queue.task_done()

    def refresh_ui(self):
        # Update inputs based on keyboard state
        self._update_keyboard_inputs()

        # EKF SLAM Predict Step
        now_time = time.time()
        dt = now_time - self.last_prediction_time
        self.last_prediction_time = now_time

        speed = 0.0
        yaw_rate = 0.0
        yaw_rate = 0.0
        speed = 0.0
        qz = 0.0
        qw = 1.0
        if self.latest_imu:
            speed = self.latest_imu.get("ground_speed_mps", 0.0)
            yaw_rate = self.latest_imu.get("imu", {}).get("angular_velocity", {}).get("z", 0.0)
            ori = self.latest_imu.get("imu", {}).get("orientation", {})
            qz = ori.get("z", 0.0)
            qw = ori.get("w", 1.0)

        # EKF SLAM Predict Step
        self.ekf.predict(speed, yaw_rate, dt)
        
        if qz != 0.0 or qw != 1.0:
            abs_theta = self.ekf.normalize_angle(2.0 * math.atan2(qz, qw))
            if not hasattr(self.ekf, 'heading_initialized'):
                self.ekf.x[2] = abs_theta
                self.ekf.heading_initialized = True
            else:
                self.ekf.update_heading(abs_theta)

        # EKF SLAM Update Step
        fused_for_ekf = []
        if not hasattr(self.ekf, 'heading_initialized'):
            self.latest_fused_measurements = None
        with self.yolo_lock:
            if self.new_fused_ready:
                fused_for_ekf = list(self.latest_fused_measurements)
                self.new_fused_ready = False

        if fused_for_ekf and speed > 0.1:
            self.ekf.update(fused_for_ekf)

        self.throttle_label.set(f"Throttle: {self.desired['throttle']:.3f}")
        self.brake_label.set(f"Brake: {self.desired['brake']:.3f}")
        self.steering_label.set(f"Steering: {self.desired['steering']:.3f}")

        mode_str = "AUTO" if self.autonomous_mode else "MANUAL"
        self.imu_status_var.set(f"81 IMU: {self.imu_status} [{mode_str}]")
        self.act_status_var.set(f"83 Actuator: {self.act_status}")
        self.vision_status_var.set(f"84 Vision: {self.vision_status}")
        self.ctrl_status_var.set(f"82 Control TX: {self.ctrl_status}")

        if self.latest_imu:
            try:
                speed_val = self.latest_imu.get("ground_speed_mps", 0.0)
                imu = self.latest_imu.get("imu", {})

                av = imu.get("angular_velocity", {})
                la = imu.get("linear_acceleration", {})
                ori = imu.get("orientation", {})

                self.speed_var.set(f"Ground Speed: {speed_val:.3f} m/s")
                self.ang_var.set(
                    f"Angular Vel: x={av.get('x', 0.0):+.4f}  y={av.get('y', 0.0):+.4f}  z={av.get('z', 0.0):+.4f}"
                )
                self.lin_var.set(
                    f"Linear Acc: x={la.get('x', 0.0):+.4f}  y={la.get('y', 0.0):+.4f}  z={la.get('z', 0.0):+.4f}"
                )
                self.ori_var.set(
                    f"Orientation: x={ori.get('x', 0.0):+.4f}  y={ori.get('y', 0.0):+.4f}  z={ori.get('z', 0.0):+.4f}  w={ori.get('w', 1.0):+.4f}"
                )
            except Exception as e:
                self.speed_var.set(f"IMU parse error: {e}")

        if self.latest_actuator:
            try:
                self.act_throttle_var.set(f"Throttle: {float(self.latest_actuator.get('throttle', 0.0)):.3f}")
                self.act_brake_var.set(f"Brake: {float(self.latest_actuator.get('brake', 0.0)):.3f}")
                self.act_steering_var.set(f"Steering: {float(self.latest_actuator.get('steering', 0.0)):.3f}")
            except Exception as e:
                self.act_throttle_var.set(f"Actuator parse error: {e}")

        # Fetch latest images and fused data under thread lock
        with self.yolo_lock:
            cam_img = self.latest_cam_img.copy() if self.latest_cam_img is not None else None
            lidar_img = self.latest_lidar_img.copy() if self.latest_lidar_img is not None else None
            fused = list(self.latest_fused_measurements) if self.latest_fused_measurements else []

        if cam_img is not None and lidar_img is not None:
            try:
                # Draw fused results on the dashboard panels
                for cone in fused:
                    label = cone["label"]
                    conf = cone["conf"]
                    cam_box = cone["cam_box"]
                    lidar_pixel = cone["lidar_pixel"]

                    # Color bounding boxes based on the actual class label (Yellow/Blue/Orange)
                    lbl = label.lower()
                    if "yellow" in lbl:
                        box_color = (255, 255, 0)  # Yellow (RGB)
                    elif "blue" in lbl:
                        box_color = (0, 120, 255)  # Blue (RGB)
                    elif "orange" in lbl:
                        box_color = (255, 165, 0)  # Orange (RGB)
                    else:
                        box_color = (255, 255, 255) # White (RGB)

                    # Green circle on the LiDAR panel still represents a valid fusion match
                    if lidar_pixel is not None:
                        r_text = f" [{cone['range']:.1f}m]"
                        cv2.circle(lidar_img, tuple(lidar_pixel), 10, (0, 255, 0), 2) # Green for LiDAR match
                    else:
                        r_text = ""

                    x1, y1, x2, y2 = cam_box
                    cv2.rectangle(cam_img, (x1, y1), (x2, y2), box_color, 2)
                    text = f"{label} {conf:.2f}{r_text}"
                    cv2.putText(cam_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Update EKF Map window if it is visible
                if hasattr(self, 'map_window') and self.map_window.winfo_viewable():
                    try:
                        ekf_img = self._draw_ekf_map()
                        disp_ekf = Image.fromarray(ekf_img)
                        tk_ekf = ImageTk.PhotoImage(disp_ekf)
                        self.map_label.configure(image=tk_ekf)
                        self.map_label.image = tk_ekf
                    except Exception as e:
                        print(f"[Dashboard] Error updating EKF map window: {e}")

                h_cam, w_cam, _ = cam_img.shape
                h_lid, w_lid, _ = lidar_img.shape

                # Combined dashboard: Camera on top, LiDAR on bottom
                dash_w = max(w_cam, w_lid)
                dash_h = h_cam + h_lid
                dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)

                # Center Camera on top
                dx_cam = (dash_w - w_cam) // 2
                dashboard[:h_cam, dx_cam:dx_cam+w_cam] = cam_img

                # Center LiDAR on bottom
                dx_lid = (dash_w - w_lid) // 2
                dashboard[h_cam:, dx_lid:dx_lid+w_lid] = lidar_img

                disp_img = Image.fromarray(dashboard)
                disp_img.thumbnail((1150, 850))
                tk_img = ImageTk.PhotoImage(disp_img)
                self.image_label.configure(image=tk_img)
                self.image_label.image = tk_img
            except Exception as e:
                self.vision_status = f"Dashboard rendering error ({e})"

        # Update EKF Measurements Text Panel
        if fused:
            lines = []
            for cone in fused[:5]:
                r_val = f"{cone['range']:.2f}m" if cone['range'] is not None else "---"
                b_deg = math.degrees(cone['bearing'])
                b_val = f"{b_deg:+.1f}°"
                lines.append(f"{cone['label'][:3].upper()}: r={r_val:<6} b={b_val}")
            if len(fused) > 5:
                lines.append(f"... and {len(fused) - 5} more")
            self.slam_text_var.set("\n".join(lines))
        else:
            self.slam_text_var.set("No detections")

        self.root.after(50, self.refresh_ui)

    def _draw_ekf_map(self):
        # Create a blank 540x540 image
        map_img = np.zeros((540, 540, 3), dtype=np.uint8)
        map_img[:] = (16, 16, 16)  # Dark background

        # Center of the map is the vehicle position
        cx, cy = 270, 270
        scale = 15.0  # pixels per meter

        xv = self.ekf.x[0]
        yv = self.ekf.x[1]
        theta = self.ekf.x[2]

        # 1. Draw scrolling grid lines
        grid_spacing = 5.0  # meters
        start_x = (int(xv / grid_spacing) - 5) * grid_spacing
        end_x = (int(xv / grid_spacing) + 5) * grid_spacing
        start_y = (int(yv / grid_spacing) - 5) * grid_spacing
        end_y = (int(yv / grid_spacing) + 5) * grid_spacing

        # Draw vertical grid lines (constant Y in global frame)
        for gy in np.arange(start_y, end_y, grid_spacing):
            dy = gy - yv
            px = int(cx - dy * scale)
            if 0 <= px < 540:
                cv2.line(map_img, (px, 0), (px, 540), (40, 40, 40), 1)

        # Draw horizontal grid lines (constant X in global frame)
        for gx in np.arange(start_x, end_x, grid_spacing):
            dx = gx - xv
            py = int(cy - dx * scale)
            if 0 <= py < 540:
                cv2.line(map_img, (0, py), (540, py), (40, 40, 40), 1)

        # Draw trajectory history (cyan line)
        if len(self.ekf.trajectory) > 1:
            pts = []
            for tx, ty in self.ekf.trajectory:
                dx = tx - xv
                dy = ty - yv
                px = int(cx - dy * scale)
                py = int(cy - dx * scale)
                pts.append((px, py))
            for i in range(len(pts) - 1):
                cv2.line(map_img, pts[i], pts[i+1], (0, 180, 255), 2)  # Cyan trajectory line

        # Draw Planned Centerline (Waypoints)
        if hasattr(self, 'controller') and self.controller.last_waypoints:
            wpt_pixels = []
            for wx, wy in self.controller.last_waypoints:
                dx = wx - xv
                dy = wy - yv
                px = int(cx - dy * scale)
                py = int(cy - dx * scale)
                wpt_pixels.append((px, py))
                cv2.circle(map_img, (px, py), 3, (255, 0, 255), -1)  # Magenta dots
            
            if len(wpt_pixels) > 1:
                for i in range(len(wpt_pixels) - 1):
                    cv2.line(map_img, wpt_pixels[i], wpt_pixels[i+1], (255, 0, 255), 1)
        
        # Draw Pure Pursuit Lookahead Target
        if hasattr(self, 'controller') and self.controller.last_target_point:
            tx, ty = self.controller.last_target_point
            dx = tx - xv
            dy = ty - yv
            px = int(cx - dy * scale)
            py = int(cy - dx * scale)
            cv2.circle(map_img, (px, py), 6, (0, 0, 255), 2)  # Red Target Reticle

        # 3. Draw mapped landmarks (cones)
        drawn_cones = 0
        for idx, l_info in enumerate(self.ekf.landmarks):
            if l_info.get("hit_count", 0) < 3:
                continue
            drawn_cones += 1
            color = l_info["color"].lower()
            lx_idx = 3 + 2 * idx
            ly_idx = 4 + 2 * idx
            if lx_idx >= len(self.ekf.x):
                continue
            lx = self.ekf.x[lx_idx]
            ly = self.ekf.x[ly_idx]

            dx = lx - xv
            dy = ly - yv
            px = int(cx - dy * scale)
            py = int(cy - dx * scale)

            if 0 <= px < 540 and 0 <= py < 540:
                # The dashboard uses RGB format (PIL Image conversion)
                if "yellow" in color:
                    c_rgb = (255, 255, 0)  # Yellow (RGB)
                elif "blue" in color:
                    c_rgb = (0, 120, 255)  # Blue (RGB)
                elif "orange" in color:
                    c_rgb = (255, 165, 0)  # Orange (RGB)
                else:
                    c_rgb = (255, 255, 255)

                cv2.circle(map_img, (px, py), 6, c_rgb, -1)
                cv2.circle(map_img, (px, py), 7, (0, 0, 0), 1)  # Outline

        # 4. Draw current vehicle pose (oriented triangle)
        car_len = 12
        car_width = 8

        # Tip point (ahead along theta)
        tx_tip = cx - int(car_len * math.sin(theta))
        ty_tip = cy - int(car_len * math.cos(theta))

        # Back corners
        tx_l = cx + int(car_len/2 * math.sin(theta)) - int(car_width * math.cos(theta))
        ty_l = cy + int(car_len/2 * math.cos(theta)) + int(car_width * math.sin(theta))

        tx_r = cx + int(car_len/2 * math.sin(theta)) + int(car_width * math.cos(theta))
        ty_r = cy + int(car_len/2 * math.cos(theta)) - int(car_width * math.sin(theta))

        pts_triangle = np.array([[tx_tip, ty_tip], [tx_l, ty_l], [tx_r, ty_r]], dtype=np.int32)
        cv2.drawContours(map_img, [pts_triangle], 0, (0, 255, 0), -1)  # Green car triangle
        cv2.drawContours(map_img, [pts_triangle], 0, (0, 0, 0), 1)  # Outline

        # Title text
        cv2.putText(map_img, f"EKF SLAM Map (Cones: {drawn_cones})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return map_img

    def _on_key_press(self, event):
        key = event.keysym.lower()
        self.pressed_keys[key] = time.time()
        
        if key == 'p':
            self.autonomous_mode = not self.autonomous_mode
            if self.autonomous_mode:
                self.using_keyboard = False
                self.steering_var.set(0.0)
                self.throttle_var.set(0.0)
                self.brake_var.set(0.0)
                self.on_slider_change()
            print(f"[Dashboard] Autonomous Mode: {self.autonomous_mode}")

    def _on_key_release(self, event):
        key = event.keysym.lower()
        release_time = time.time()
        self.root.after(20, lambda: self._confirm_release(key, release_time))

    def _confirm_release(self, key, release_time):
        if key in self.pressed_keys and self.pressed_keys[key] > release_time:
            return
        if key in self.pressed_keys:
            del self.pressed_keys[key]

    def _update_keyboard_inputs(self):
        drive_keys = {"left", "right", "up", "down", "a", "d", "w", "s", "space"}
        active_drive_keys = any(k in self.pressed_keys for k in drive_keys)

        if active_drive_keys:
            self.using_keyboard = True

        if not self.using_keyboard:
            return

        current_steering = self.steering_var.get()
        current_throttle = self.throttle_var.get()
        current_brake = self.brake_var.get()

        steer_left = "left" in self.pressed_keys or "a" in self.pressed_keys
        steer_right = "right" in self.pressed_keys or "d" in self.pressed_keys
        throttle_up = "up" in self.pressed_keys or "w" in self.pressed_keys
        brake_down = "down" in self.pressed_keys or "s" in self.pressed_keys
        space_brake = "space" in self.pressed_keys

        changed = False

        # Steering accumulation and decay (20Hz loop, dt=50ms)
        if steer_left and not steer_right:
            new_steering = max(-1.0, current_steering - 0.08)
            if new_steering != current_steering:
                current_steering = new_steering
                changed = True
        elif steer_right and not steer_left:
            new_steering = min(1.0, current_steering + 0.08)
            if new_steering != current_steering:
                current_steering = new_steering
                changed = True
        else:
            # Decay steering back to 0.0
            if current_steering > 0.0:
                current_steering = max(0.0, current_steering - 0.1)
                changed = True
            elif current_steering < 0.0:
                current_steering = min(0.0, current_steering + 0.1)
                changed = True

        # Throttle accumulation and decay
        if throttle_up:
            # Limit to 0.40 for safe testing/driving
            new_throttle = min(0.40, current_throttle + 0.05)
            if new_throttle != current_throttle:
                current_throttle = new_throttle
                changed = True
        else:
            if current_throttle > 0.0:
                current_throttle = max(0.0, current_throttle - 0.08)
                changed = True

        # Brake accumulation and decay
        if space_brake:
            if current_brake != 1.0 or current_throttle != 0.0:
                current_brake = 1.0
                current_throttle = 0.0
                changed = True
        elif brake_down:
            new_brake = min(1.0, current_brake + 0.15)
            if new_brake != current_brake or current_throttle != 0.0:
                current_brake = new_brake
                current_throttle = 0.0
                changed = True
        else:
            if current_brake > 0.0:
                current_brake = max(0.0, current_brake - 0.15)
                changed = True

        if changed:
            self.steering_var.set(round(current_steering, 3))
            self.throttle_var.set(round(current_throttle, 3))
            self.brake_var.set(round(current_brake, 3))
            self.on_slider_change()

        # Turn off using_keyboard state if everything is fully zeroed out and inactive
        if not active_drive_keys and round(current_steering, 3) == 0.0 and round(current_throttle, 3) == 0.0 and round(current_brake, 3) == 0.0:
            self.using_keyboard = False

    def on_close(self):
        try:
            self.ctrl_tx.send_json_line({
                "throttle": 0.0,
                "brake": 1.0,
                "steering": 0.0
            })
        except Exception:
            pass

        self.imu_rx.close()
        self.act_rx.close()
        self.vision_rx.close()
        self.ctrl_tx.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TestConsoleApp(root)
    root.mainloop()
