import sys
import threading
import types
import unittest
from collections import defaultdict, deque


# Import dashboard logic without loading the real YOLO package or model.
fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = object
sys.modules.setdefault("ultralytics", fake_ultralytics)

from ekf_slam import EKFSLAM
from test import PENDING_COLOR_MAX, TestConsoleApp


class ValueStub:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ButtonStub:
    def configure(self, **kwargs):
        self.options = kwargs


class MappingLogicTests(unittest.TestCase):
    def make_app(self):
        app = TestConsoleApp.__new__(TestConsoleApp)
        app.ekf = EKFSLAM()
        app.ekf.heading_initialized = True
        app.latest_imu = None
        app.cone_candidates = []
        app.candidate_scan_index = 0
        app.next_candidate_id = 0
        app.last_geometry_promotion_scan = 0
        app.initial_map_ready = False
        app.map_readiness_reason = "Waiting"
        app.map_x_min_m = -15.0
        app.map_x_max_m = 15.0
        app.map_y_min_m = -15.0
        app.map_y_max_m = 15.0
        app.pose_history = deque(maxlen=200)
        app.metrics = defaultdict(float)
        app.sensor_lock = threading.Lock()
        app.lidar_buffer = deque(maxlen=60)
        app.pending_color_results = deque()
        app.yolo_lock = threading.Lock()
        return app

    @staticmethod
    def observation(color="blue", box_height=12.0, bearing=0.2):
        return [{
            "range": 6.0,
            "bearing": bearing,
            "color": color,
            "box_height": box_height,
            "conf": 0.90,
            "sync_delta_ns": 10_000_000,
            "projection_error_px": 1.0,
            "projection_margin_px": 10.0,
            "lidar_timestamp_ns": 0,
        }]

    def test_geometry_becomes_unknown_then_receives_colour(self):
        app = self.make_app()

        self.assertEqual(app._mapping_measurements(self.observation(bearing=0.19)), [])
        self.assertEqual(app._mapping_measurements(self.observation(bearing=0.20)), [])
        accepted = app._mapping_measurements(self.observation(bearing=0.21))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["color"], "unknown")
        self.assertTrue(app.cone_candidates[0]["geometry_confirmed"])
        app.ekf.update(accepted)

        # Weighted close observations recolour the same candidate/landmark.
        for _ in range(3):
            app._apply_color_measurements(
                self.observation(color="blue", box_height=22.0)
            )
            if app.cone_candidates[0]["color"] == "blue":
                break
        self.assertEqual(app.cone_candidates[0]["color"], "blue")
        self.assertEqual(len(app.ekf.landmarks), 1)
        self.assertEqual(app.ekf.landmarks[0]["color"], "blue")

    def test_adaptive_world_coordinate_conversion(self):
        app = self.make_app()
        self.assertEqual(app._world_to_map_pixel(0.0, 0.0), (350, 350))
        self.assertEqual(app._world_to_map_pixel(-15.0, -15.0), (45, 655))
        self.assertEqual(app._world_to_map_pixel(15.0, 15.0), (655, 45))

    def test_map_expands_for_car_and_never_shrinks(self):
        app = self.make_app()
        app.ekf.x[0] = 20.0
        app._update_map_bounds()
        expanded_bounds = (
            app.map_x_min_m, app.map_x_max_m,
            app.map_y_min_m, app.map_y_max_m,
        )
        self.assertGreaterEqual(app.map_x_max_m, 25.0)

        app.ekf.x[0] = 0.0
        app._update_map_bounds()
        self.assertEqual(expanded_bounds, (
            app.map_x_min_m, app.map_x_max_m,
            app.map_y_min_m, app.map_y_max_m,
        ))

    def test_provisional_outlier_does_not_expand_map(self):
        app = self.make_app()
        app.cone_candidates = [{
            "x": 1000.0,
            "y": 1000.0,
            "geometry_confirmed": False,
            "position_uncertainty": 1.0,
        }]
        app._update_map_bounds()
        self.assertEqual((app.map_x_min_m, app.map_x_max_m), (-15.0, 15.0))

    def test_pose_is_interpolated_and_briefly_extrapolated(self):
        app = self.make_app()
        app.pose_history.extend([
            (1_000_000_000, 0.0, 0.0, 0.0, 2.0, 0.0),
            (2_000_000_000, 2.0, 0.0, 0.0, 2.0, 0.0),
        ])
        pose, _ = app._pose_at(1_500_000_000)
        self.assertAlmostEqual(pose[0], 1.0)
        future_pose, _ = app._pose_at(2_050_000_000)
        self.assertAlmostEqual(future_pose[0], 2.1)

    def test_lidar_geometry_survives_without_camera_match(self):
        app = self.make_app()
        observations = app._lidar_observations(
            [(350, 400, 4.0, 0.0, 4.0)], 1_000_000_000,
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["color"], "unknown")
        self.assertIsNone(observations[0]["cam_box"])

    def test_close_colour_has_more_weight_than_distant_colour(self):
        close = self.observation(box_height=30.0)[0]
        close["range"] = 2.0
        far = dict(close, range=7.5, box_height=8.0)
        self.assertGreater(
            TestConsoleApp._color_weight(close),
            TestConsoleApp._color_weight(far),
        )

    @staticmethod
    def candidate(identifier, x, y, color=None, source="unknown", orange_score=0.0):
        return {
            "id": identifier, "x": x, "y": y,
            "geometry_confirmed": True, "color_confirmed": color is not None,
            "color": color, "color_source": source,
            "color_scores": {"blue": 0.0, "yellow": 0.0, "orange": orange_score},
        }

    def test_passed_boundary_gap_is_inferred(self):
        app = self.make_app()
        app.ekf.x[0] = 10.0
        unknown = self.candidate(0, 0.0, 0.0)
        app.cone_candidates = [
            unknown,
            self.candidate(1, -2.0, 0.0, "yellow", "vision"),
            self.candidate(2, 2.0, 0.0, "yellow", "vision"),
        ]
        app._infer_missing_colors()
        self.assertEqual((unknown["color"], unknown["color_source"]), ("yellow", "boundary"))

    def test_orange_rectangle_propagates_only_with_evidence(self):
        app = self.make_app()
        app.ekf.x[0] = 10.0
        group = [
            self.candidate(0, 0.0, -2.0, "orange", "vision", 0.3),
            self.candidate(1, 0.0, 2.0, "orange", "vision", 0.3),
            self.candidate(2, 2.0, -2.0),
            self.candidate(3, 2.0, 2.0),
        ]
        app.cone_candidates = group
        app._infer_missing_colors()
        self.assertTrue(all(candidate["color"] == "orange" for candidate in group))

    def test_nearest_lidar_is_selected_by_timestamp(self):
        app = self.make_app()
        app.lidar_buffer.extend([(1_000_000_000, ["old"]), (1_040_000_000, ["near"])])
        self.assertEqual(app._nearest_lidar(1_035_000_000)[1], ["near"])

    def test_pending_colour_waits_for_future_lidar(self):
        entry = {"camera_timestamp": 1_070_000_000, "deadline": 2.0}
        finished, nearest = TestConsoleApp._pending_lidar_match(
            entry, [(1_000_000_000, ["old"])], now=1.0,
        )
        self.assertFalse(finished)
        self.assertIsNone(nearest)

    def test_future_lidar_rescues_pending_colour(self):
        entry = {"camera_timestamp": 1_070_000_000, "deadline": 2.0}
        finished, nearest = TestConsoleApp._pending_lidar_match(
            entry,
            [(1_000_000_000, ["old"]), (1_100_000_000, ["future"])],
            now=1.0,
        )
        self.assertTrue(finished)
        self.assertEqual(nearest[1], ["future"])

    def test_pending_colour_expires_without_valid_lidar(self):
        entry = {"camera_timestamp": 1_070_000_000, "deadline": 2.0}
        finished, nearest = TestConsoleApp._pending_lidar_match(
            entry, [(1_000_000_000, ["old"])], now=2.1,
        )
        self.assertTrue(finished)
        self.assertIsNone(nearest)

    def test_pending_colour_queue_is_bounded(self):
        app = self.make_app()
        for timestamp in range(PENDING_COLOR_MAX + 2):
            app._queue_pending_color([], timestamp + 1, 640, 0.0, 0.0)
        self.assertEqual(len(app.pending_color_results), PENDING_COLOR_MAX)
        self.assertEqual(app.metrics["sync_rejects"], 2)

    def test_fixed_map_renders_provisional_and_moving_car(self):
        app = self.make_app()
        app.autonomous_mode = False
        app.controller = types.SimpleNamespace(last_waypoints=[], last_target_point=None)
        app.cone_candidates = [{
            "x": 5.0,
            "y": 2.0,
            "geometry_confirmed": False,
            "position_uncertainty": 0.2,
        }]
        app.ekf.x[0] = 3.0
        app.ekf.x[1] = -4.0
        image = app._draw_ekf_map()
        self.assertEqual(image.shape, (700, 700, 3))
        self.assertEqual(app._world_to_map_pixel(3.0, -4.0), (411, 431))

    def test_button_and_p_key_use_same_autonomy_toggle(self):
        app = self.make_app()
        app.autonomous_mode = False
        app.using_keyboard = False
        app.steering_var = ValueStub(0.0)
        app.throttle_var = ValueStub(0.0)
        app.brake_var = ValueStub(1.0)
        app.autonomy_button_text = ValueStub("Enable Autonomy")
        app.autonomy_button = ButtonStub()
        app.manual_desired = {"throttle": 0.0, "brake": 1.0, "steering": 0.0}
        app.pressed_keys = {}
        app.on_slider_change = lambda *_: None
        app._set_desired = lambda *_: None

        app.toggle_autonomy()  # UI button command uses this method.
        self.assertTrue(app.autonomous_mode)
        self.assertEqual(app.autonomy_button_text.get(), "Disable Autonomy")

        event = types.SimpleNamespace(keysym="p")
        app._on_key_press(event)
        self.assertFalse(app.autonomous_mode)
        self.assertEqual(app.autonomy_button_text.get(), "Enable Autonomy")


if __name__ == "__main__":
    unittest.main()
