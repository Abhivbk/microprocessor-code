import mmap
import os
import struct
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds


VEHICLE_NAME = "FSCar"
UPDATE_INTERVAL_SECONDS = 0.05

SHARED_MEM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground")
)
ACTUATOR_BIN_PATH = os.path.join(SHARED_MEM_DIR, "abs_current.bin")

BINARY_FORMAT = "<Qfff"
STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print("[actuator_state_node] Connected to FSDS", flush=True)
            return client
        except Exception as exc:
            print(f"[actuator_state_node] FSDS not ready: {exc}", flush=True)
            time.sleep(2)


def publish_actuator_state():
    os.makedirs(SHARED_MEM_DIR, exist_ok=True)
    if not os.path.exists(ACTUATOR_BIN_PATH) or os.path.getsize(ACTUATOR_BIN_PATH) != STRUCT_SIZE:
        with open(ACTUATOR_BIN_PATH, "wb") as actuator_file:
            actuator_file.write(b"\x00" * STRUCT_SIZE)

    client = connect_fsds_forever()
    with open(ACTUATOR_BIN_PATH, "r+b") as actuator_file:
        ram = mmap.mmap(actuator_file.fileno(), STRUCT_SIZE)
        print("[actuator_state_node] Publishing actuator feedback at 20 Hz", flush=True)
        try:
            while True:
                try:
                    controls = client.getCarControls(VEHICLE_NAME)
                    packed = struct.pack(
                        BINARY_FORMAT,
                        int(time.time() * 1000),
                        float(controls.throttle),
                        float(controls.brake),
                        float(controls.steering),
                    )
                    ram[0:STRUCT_SIZE] = packed
                    time.sleep(UPDATE_INTERVAL_SECONDS)
                except Exception as exc:
                    print(f"[actuator_state_node] Read failed: {exc}", flush=True)
                    client = connect_fsds_forever()
        finally:
            ram.close()


if __name__ == "__main__":
    try:
        publish_actuator_state()
    except KeyboardInterrupt:
        print("[actuator_state_node] Stopped", flush=True)
