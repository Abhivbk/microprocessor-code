import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

VEHICLE_NAME = "FSCar"
LIDAR_NAME = "Lidar1"

LIDAR_W = 700
LIDAR_H = 540
LIDAR_RANGE_METERS = 20.0
LIDAR_CLUSTER_DIST = 0.30
LIDAR_MIN_CLUSTER_POINTS = 2
LIDAR_MAX_CLUSTER_POINTS = 60
MAX_CONE_SPREAD_X = 0.50
MAX_CONE_SPREAD_Y = 0.50

def cluster_lidar_points(points_xy, cluster_dist=LIDAR_CLUSTER_DIST):
    """Label 2-D points connected by neighbour distances below cluster_dist."""
    points = np.asarray(points_xy, dtype=np.float32)
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)

    # The KD-tree finds only nearby pairs; SciPy groups their connected graph.
    pairs = cKDTree(points).query_pairs(cluster_dist, output_type="ndarray")
    if pairs.size:
        deltas = points[pairs[:, 0]] - points[pairs[:, 1]]
        pairs = pairs[np.einsum("ij,ij->i", deltas, deltas) < cluster_dist**2]
    if pairs.size == 0:
        return np.arange(len(points), dtype=np.int32)

    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(points), len(points)),
    ).tocsr()
    return connected_components(graph, directed=False, return_labels=True)[1]

def detect_cones_lidar(client, profile=None):
    """
    Processes active lidar point cloud data to filter, cluster, and detect cones.
    Returns:
        roi (Nx2 ndarray): filtered point-cloud points in the region of interest
        cones (list of (cx, cy) tuples): coordinates of detected cone centers
        timestamp_ns (int): FSDS timestamp of the LiDAR scan
    """
    rpc_started = time.perf_counter()
    lidar = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    if profile is not None:
        profile["rpc_s"] = time.perf_counter() - rpc_started

    processing_started = time.perf_counter()
    if len(lidar.point_cloud) < 3:
        if profile is not None:
            profile["processing_s"] = time.perf_counter() - processing_started
        return [], [], int(lidar.time_stamp)

    pts = np.asarray(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
    mask = (
        (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= LIDAR_RANGE_METERS)
        & (np.abs(pts[:, 1]) <= 10.0)
        & (pts[:, 2] >= -1.5)
        & (pts[:, 2] <= 1.0)
    )
    roi = pts[mask, :2]

    if len(roi) == 0:
        if profile is not None:
            profile["processing_s"] = time.perf_counter() - processing_started
        return [], [], int(lidar.time_stamp)

    labels = cluster_lidar_points(roi, LIDAR_CLUSTER_DIST)

    # Sorting once lets NumPy calculate every cluster's statistics in bulk.
    order = np.argsort(labels, kind="stable")
    sorted_points, sorted_labels = roi[order], labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_points)])
    centres = np.add.reduceat(sorted_points, starts) / counts[:, None]
    spreads = (
        np.maximum.reduceat(sorted_points, starts)
        - np.minimum.reduceat(sorted_points, starts)
    )
    valid = (
        (counts >= LIDAR_MIN_CLUSTER_POINTS)
        & (counts <= LIDAR_MAX_CLUSTER_POINTS)
        & (spreads[:, 0] <= MAX_CONE_SPREAD_X)
        & (spreads[:, 1] <= MAX_CONE_SPREAD_Y)
    )
    cones = [tuple(map(float, centre)) for centre in centres[valid]]

    if profile is not None:
        profile["processing_s"] = time.perf_counter() - processing_started
    return roi, cones, int(lidar.time_stamp)
