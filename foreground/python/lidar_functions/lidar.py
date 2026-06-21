import math
import numpy as np

VEHICLE_NAME = "FSCar"
LIDAR_NAME = "Lidar"

LIDAR_W = 700
LIDAR_H = 540
LIDAR_RANGE_METERS = 20.0
LIDAR_CLUSTER_DIST = 0.30
LIDAR_MIN_CLUSTER_POINTS = 1
LIDAR_MAX_CLUSTER_POINTS = 60
MAX_CONE_SPREAD_X = 1.2
MAX_CONE_SPREAD_Y = 1.2

def euclidean_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def cluster_lidar_points(points_xy, cluster_dist=LIDAR_CLUSTER_DIST):
    clusters = []
    used = np.zeros(len(points_xy), dtype=bool)

    for i in range(len(points_xy)):
        if used[i]:
            continue

        cluster = [points_xy[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(points_xy)):
                if used[j]:
                    continue
                for cp in cluster:
                    if euclidean_2d(points_xy[j], cp) < cluster_dist:
                        cluster.append(points_xy[j])
                        used[j] = True
                        changed = True
                        break

        clusters.append(cluster)

    return clusters

def detect_cones_lidar(client):
    """
    Processes active lidar point cloud data to filter, cluster, and detect cones.
    Returns:
        roi (list of (x, y) tuples): filtered point cloud points in the Region of Interest
        cones (list of (cx, cy) tuples): coordinates of detected cone centers
    """
    lidar = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    if len(lidar.point_cloud) < 3:
        return [], []

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

    if not roi:
        return [], []

    clusters = cluster_lidar_points(roi, LIDAR_CLUSTER_DIST)

    cones = []

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

        if spread_x > MAX_CONE_SPREAD_X or spread_y > MAX_CONE_SPREAD_Y:
            continue

        cones.append((cx, cy))

    return roi, cones
