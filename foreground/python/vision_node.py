import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from cam_functions.camera import get_camera_frame
from cam_functions.shared_mem import save_to_shared_memory as save_camera
from lidar_functions.lidar import detect_cones_lidar
from lidar_functions.shared_mem import save_to_shared_memory as save_lidar


class RateLog:
    """Report averaged sensor timings once per second."""

    def __init__(self, name):
        self.name = name
        self._reset()

    def _reset(self):
        self.started = time.perf_counter()
        self.count = self.unique_count = 0
        self.rpc_s = self.processing_s = self.shm_s = self.work_s = 0.0
        self.first_timestamp = self.last_timestamp = 0

    def add(self, timestamp, profile, shm_s, work_s, unique=True):
        self.count += 1
        self.unique_count += int(unique)
        self.rpc_s += profile.get("rpc_s", 0.0)
        self.processing_s += profile.get("processing_s", 0.0)
        self.shm_s += shm_s
        self.work_s += work_s
        if unique and timestamp > 0:
            self.first_timestamp = self.first_timestamp or timestamp
            self.last_timestamp = timestamp

        elapsed = time.perf_counter() - self.started
        if elapsed < 1.0:
            return
        sensor_elapsed = (self.last_timestamp - self.first_timestamp) / 1e9
        sensor_hz = (self.unique_count - 1) / sensor_elapsed if sensor_elapsed > 0 else 0.0
        ms_per_sample = 1000.0 / max(1, self.count)
        print(
            f"[profile:{self.name}] poll={self.count / elapsed:.1f}fps "
            f"unique={self.unique_count / elapsed:.1f}fps sensor={sensor_hz:.1f}Hz "
            f"rpc={self.rpc_s * ms_per_sample:.1f}ms "
            f"process={self.processing_s * ms_per_sample:.1f}ms "
            f"shm={self.shm_s * ms_per_sample:.1f}ms "
            f"work={self.work_s * ms_per_sample:.1f}ms",
            flush=True,
        )
        self._reset()


def connect(name):
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print(f"[vision_node] {name} connected", flush=True)
            return client
        except Exception as error:
            print(f"[vision_node] {name} waiting: {error}", flush=True)
            time.sleep(2)


def camera_worker(stop):
    client, sequence = connect("camera"), 0
    rate = RateLog("camera")
    while not stop.is_set():
        started = time.perf_counter()
        try:
            profile = {}
            frame, timestamp = get_camera_frame(client, profile)
            if frame is not None:
                sequence += 1
                shm_started = time.perf_counter()
                save_camera(frame, timestamp, sequence)
                shm_s = time.perf_counter() - shm_started
                rate.add(timestamp, profile, shm_s, time.perf_counter() - started)
        except Exception as error:
            print(f"[vision_node] camera error: {error}", flush=True)
            client = connect("camera")
        stop.wait(max(0.001, 0.033 - (time.perf_counter() - started)))


def lidar_worker(stop):
    client, sequence = connect("LiDAR"), 0
    rate = RateLog("lidar")
    last_timestamp = None
    while not stop.is_set():
        started = time.perf_counter()
        try:
            profile = {}
            _, cones, timestamp = detect_cones_lidar(client, profile)
            unique = timestamp > 0 and timestamp != last_timestamp
            shm_s = 0.0
            if unique:
                last_timestamp = timestamp
                sequence += 1
                shm_started = time.perf_counter()
                save_lidar(cones, timestamp, sequence)
                shm_s = time.perf_counter() - shm_started
            rate.add(
                timestamp, profile, shm_s,
                time.perf_counter() - started, unique,
            )
        except Exception as error:
            print(f"[vision_node] LiDAR error: {error}", flush=True)
            client = connect("LiDAR")
        stop.wait(max(0.001, 0.033 - (time.perf_counter() - started)))


if __name__ == "__main__":
    print("[vision_node] Independent timestamped camera and LiDAR streams", flush=True)
    stop_event = threading.Event()
    workers = [
        threading.Thread(target=camera_worker, args=(stop_event,), daemon=True),
        threading.Thread(target=lidar_worker, args=(stop_event,), daemon=True),
    ]
    for worker in workers:
        worker.start()
    try:
        while all(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("[vision_node] Stopped", flush=True)
