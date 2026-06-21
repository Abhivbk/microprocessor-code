import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from cam_functions.camera import get_camera_frame
from cam_functions.shared_mem import save_to_shared_memory as save_cam_shm
from lidar_functions.lidar import detect_cones_lidar
from lidar_functions.shared_mem import save_to_shared_memory as save_lidar_shm

def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print("[vision_node] Connected to FSDS", flush=True)
            return client
        except Exception as e:
            print(f"[vision_node] FSDS not ready: {e}", flush=True)
            time.sleep(2)

if __name__ == "__main__":
    print("[vision_node] Running with user's cam and lidar functions (mmap).", flush=True)
    client = connect_fsds_forever()
    try:
        while True:
            start_time = time.time()

            try:
                frame = get_camera_frame(client)
                if frame is not None:
                    save_cam_shm(frame)
                
                roi, cones = detect_cones_lidar(client)
                save_lidar_shm(cones)

            except Exception as e:
                print(f"Error: {e}")
                client = connect_fsds_forever()

            elapsed = time.time() - start_time
            sleep_time = max(0.005, 0.033 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print("[vision_node] Stopped", flush=True)