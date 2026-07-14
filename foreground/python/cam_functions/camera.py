import cv2
import numpy as np
import fsds
import time

VEHICLE_NAME = "FSCar"
CAMERA_NAME = "cam1"
CAM_W = 960
CAM_H = 540
MIN_CONTOUR_AREA = 120

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 120, 0)
ORANGE = (0, 165, 255)

def get_camera_frame(client, profile=None):
    rpc_started = time.perf_counter()
    responses = client.simGetImages([
        fsds.ImageRequest(
            camera_name=CAMERA_NAME,
            image_type=fsds.ImageType.Scene,
            pixels_as_float=False,
            compress=False
        )
    ], vehicle_name=VEHICLE_NAME)
    if profile is not None:
        profile["rpc_s"] = time.perf_counter() - rpc_started

    processing_started = time.perf_counter()

    if not responses:
        if profile is not None:
            profile["processing_s"] = time.perf_counter() - processing_started
        return None, 0

    img = responses[0]
    if img.width == 0 or img.height == 0:
        if profile is not None:
            profile["processing_s"] = time.perf_counter() - processing_started
        return None, 0

    arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
    frame = arr.reshape(img.height, img.width, 3)
    if profile is not None:
        profile["processing_s"] = time.perf_counter() - processing_started

    return frame, int(img.time_stamp)


def blank_camera_panel():
    """
    Generates a default blank camera panel with standard 'No camera feed' label.
    """
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(img, "No camera feed", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    return img
