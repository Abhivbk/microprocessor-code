import sys
import os
import math
import time
import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
CAMERA_NAME = "FrontCam"
LIDAR_NAME = "Lidar"

PORT = 84

CAM_W = 960
CAM_H = 540
LIDAR_W = 700
LIDAR_H = 540
DASH_W = CAM_W + LIDAR_W
DASH_H = 540

LIDAR_RANGE_METERS = 20.0
LIDAR_CLUSTER_DIST = 0.35
LIDAR_MIN_CLUSTER_POINTS = 3
LIDAR_MAX_CLUSTER_POINTS = 60
MIN_CONTOUR_AREA = 120

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 120, 0)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
GRAY = (60, 60, 60)


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print("[vision_node] Connected to FSDS")
            return client
        except Exception as e:
            print(f"[vision_node] FSDS not ready: {e}")
            time.sleep(2)


client = connect_fsds_forever()

tcp = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT)
tcp.start()


def get_camera_frame():
    global client
    try:
        responses = client.simGetImages(
            [
                fsds.ImageRequest(
                    camera_name=CAMERA_NAME,
                    image_type=fsds.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False
                )
            ],
            vehicle_name=VEHICLE_NAME
        )

        if not responses:
            return None

        img = responses[0]
        if img.width == 0 or img.height == 0:
            return None

        arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        frame = arr.reshape(img.height, img.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    except Exception:
        client = connect_fsds_forever()
        return None


def detect_cones_camera(frame):
    annotated = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = {
        "yellow": cv2.inRange(hsv, np.array([18, 100, 80]), np.array([40, 255, 255])),
        "blue":   cv2.inRange(hsv, np.array([90, 100, 60]), np.array([130, 255, 255])),
        "orange": cv2.inRange(hsv, np.array([5, 120, 80]), np.array([18, 255, 255])),
    }

    detections = []

    for color_name, mask in masks.items():
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)
            if aspect < 0.8 or h < 12:
                continue

            if color_name == "yellow":
                box_color = YELLOW
            elif color_name == "blue":
                box_color = BLUE
            else:
                box_color = ORANGE

            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated,
                f"{color_name} cone",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )
            detections.append({"x": x, "y": y, "w": w, "h": h, "color": color_name})

    cv2.putText(
        annotated,
        f"Camera cones: {len(detections)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        GREEN,
        2
    )

    return annotated


def cluster_lidar_points(points_xy, cluster_dist=LIDAR_CLUSTER_DIST):
    if len(points_xy) == 0:
        return []
    if len(points_xy) == 1:
        return [points_xy]

    pts = np.array(points_xy)
    labels = fclusterdata(pts, t=cluster_dist, criterion='distance', method='single')

    clusters = {}
    for pt, label in zip(points_xy, labels):
        clusters.setdefault(label, []).append(pt)

    return list(clusters.values())


def detect_cones_lidar():
    global client

    panel = np.zeros((LIDAR_H, LIDAR_W, 3), dtype=np.uint8)
    center_x = LIDAR_W // 2
    center_y = LIDAR_H - 40
    scale = (LIDAR_H - 80) / LIDAR_RANGE_METERS

    for d in range(0, int(LIDAR_RANGE_METERS) + 1, 5):
        py = int(center_y - d * scale)
        cv2.line(panel, (0, py), (LIDAR_W, py), GRAY, 1)
        cv2.putText(panel, f"{d}m", (10, max(15, py - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    for lateral in range(-20, 21, 5):
        px = int(center_x + lateral * scale)
        cv2.line(panel, (px, 0), (px, LIDAR_H), GRAY, 1)

    cv2.circle(panel, (center_x, center_y), 6, CYAN, -1)
    cv2.putText(panel, "Car", (center_x + 10, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

    try:
        lidar = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
        if len(lidar.point_cloud) < 3:
            cv2.putText(panel, "No lidar points", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            return panel

        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

        roi = []
        for p in pts:
            x, y, z = float(p[0]), float(p[1]), float(p[2])

            if x < 0.0 or x > LIDAR_RANGE_METERS:
                continue
            if abs(y) > 10.0:
                continue
            if z < -1.5 or z > 1.0:
                continue

            roi.append((x, y))
            px = int(center_x + y * scale)
            py = int(center_y - x * scale)
            if 0 <= px < LIDAR_W and 0 <= py < LIDAR_H:
                cv2.circle(panel, (px, py), 1, GREEN, -1)

        if not roi:
            return panel

        clusters = cluster_lidar_points(roi, LIDAR_CLUSTER_DIST)

        cone_count = 0

        for cluster in clusters:
            n = len(cluster)
            if n < LIDAR_MIN_CLUSTER_POINTS or n > LIDAR_MAX_CLUSTER_POINTS:
                continue

            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            cx = sum(xs) / n
            cy = sum(ys) / n

            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)

            if spread_x > 0.8 or spread_y > 0.8:
                continue

            px = int(center_x + cy * scale)
            py = int(center_y - cx * scale)
            cv2.circle(panel, (px, py), 7, ORANGE, 2)
            cone_count += 1

        cv2.putText(panel, f"Lidar cones: {cone_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        return panel

    except Exception:
        client = connect_fsds_forever()
        cv2.putText(panel, "Lidar reconnecting...", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        return panel


def blank_camera_panel():
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(img, "No camera feed", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    return img


def build_dashboard(cam_panel, lidar_panel):
    dash = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    dash[:, :CAM_W] = cam_panel
    dash[:, CAM_W:CAM_W + LIDAR_W] = lidar_panel
    return dash


if __name__ == "__main__":
    print("[vision_node] Running in headless mode")
    try:
        while True:
            start_time = time.time()

            frame = get_camera_frame()
            if frame is None:
                cam_panel = blank_camera_panel()
            else:
                cam_panel = detect_cones_camera(frame)
                cam_panel = cv2.resize(cam_panel, (CAM_W, CAM_H))

            lidar_panel = detect_cones_lidar()
            dashboard = build_dashboard(cam_panel, lidar_panel)

            ok, encoded = cv2.imencode(".jpg", dashboard, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                tcp.send_bytes(encoded.tobytes())

            # Maintain ~30 FPS loop rate (approx 33ms)
            elapsed = time.time() - start_time
            sleep_time = max(0.005, 0.033 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        tcp.stop()
        print("[vision_node] Stopped")