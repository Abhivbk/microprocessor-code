import os
import time
import json
import math
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

SHARED_MEM_DIR_FG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sharedmemory", "foreground"))
SHARED_MEM_DIR_BG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sharedmemory", "background"))
LIDAR_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "lidar.bin")
LIDAR_OUT_PATH = os.path.join(SHARED_MEM_DIR_BG, "lidar_cones.json")

LIDAR_RANGE_METERS = 20.0
LIDAR_CLUSTER_DIST = 0.35
LIDAR_MIN_CLUSTER_POINTS = 3
LIDAR_MAX_CLUSTER_POINTS = 60

LIDAR_W, LIDAR_H = 700, 540
center_x = LIDAR_W // 2
center_y = LIDAR_H - 40
scale = (LIDAR_H - 80) / LIDAR_RANGE_METERS

def atomic_write_json(file_path, data):
    tmp_path = file_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, file_path)
    except Exception as e:
        pass

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

def process_lidar():
    if not os.path.exists(LIDAR_BIN_PATH):
        return
        
    try:
        raw_bytes = np.fromfile(LIDAR_BIN_PATH, dtype=np.float32)
        if len(raw_bytes) < 3:
            return
        pts = raw_bytes.reshape(-1, 3)
    except Exception:
        return

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

    if not roi:
        atomic_write_json(LIDAR_OUT_PATH, [])
        return

    clusters = cluster_lidar_points(roi, LIDAR_CLUSTER_DIST)
    cone_candidates = []

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

        dist = math.sqrt(cx*cx + cy*cy)
        px = int(center_x + cy * scale)
        py = int(center_y - cx * scale)
        
        cone_candidates.append({
            "cx": px,
            "cy": py,
            "dist": dist,
            "lat_m": cy,
            "fwd_m": cx,
        })

    atomic_write_json(LIDAR_OUT_PATH, cone_candidates)

if __name__ == "__main__":
    print("[lidar_cone_detection] Running...", flush=True)
    os.makedirs(SHARED_MEM_DIR_BG, exist_ok=True)
    try:
        while True:
            process_lidar()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[lidar_cone_detection] Stopped.")
