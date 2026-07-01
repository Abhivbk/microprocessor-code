import sys
import os
import time
import mmap
import struct
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

VEHICLE_NAME = "FSCar"

SHARED_MEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "background"))
CONTROLS_FILE = os.path.join(SHARED_MEM_DIR, "control_instruction.bin")

BINARY_FORMAT = "<Qfff"
STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            client.enableApiControl(True, VEHICLE_NAME)
            print("[control_input_node] Connected to FSDS", flush=True)
            return client
        except Exception as e:
            print(f"[control_input_node] FSDS not ready: {e}", flush=True)
            time.sleep(2)

client = connect_fsds_forever()
car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 1.0
car_controls.steering = 0.0

client_lock = threading.Lock()

latest = {
    "throttle": 0.0,
    "brake": 1.0,
    "steering": 0.0,
    "timestamp_ms": int(time.time() * 1000),
}

def apply_controls(throttle, brake, steering):
    global client

    with client_lock:
        car_controls.throttle = throttle
        car_controls.brake = brake
        car_controls.steering = steering

        try:
            client.setCarControls(car_controls, VEHICLE_NAME)
        except Exception as e:
            print(f"[control_input_node] setCarControls failed: {e}")
            client = connect_fsds_forever()
            try:
                client.setCarControls(car_controls, VEHICLE_NAME)
            except Exception:
                pass

def watchdog_and_control_loop():
    print("[control_input_node] Control loop started", flush=True)
    os.makedirs(SHARED_MEM_DIR, exist_ok=True)
    
    if not os.path.exists(CONTROLS_FILE) or os.path.getsize(CONTROLS_FILE) != STRUCT_SIZE:
        with open(CONTROLS_FILE, "wb") as f:
            f.write(struct.pack(BINARY_FORMAT, int(time.time() * 1000), 0.0, 1.0, 0.0))

    with open(CONTROLS_FILE, "r+b") as f:
        ram = mmap.mmap(f.fileno(), STRUCT_SIZE)

        while True:
            time.sleep(0.05)
            now = int(time.time() * 1000)
            
            raw_data = ram[0:STRUCT_SIZE]
            shm_time, shm_throttle, shm_brake, shm_steering = struct.unpack(BINARY_FORMAT, raw_data)

            is_moving = (latest["throttle"] > 0.0 or latest["brake"] < 1.0)
            
            # Watchdog: If the timestamp in shared memory is more than 500ms old, the background process died/froze
            if now - shm_time > 500:
                if is_moving:
                    print("[control_input_node] WATCHDOG TRIGGERED: Active Braking due to packet timeout.", flush=True)
                    apply_controls(0.0, 1.0, 0.0)
                    latest["throttle"] = 0.0
                    latest["brake"] = 1.0
                    latest["steering"] = 0.0
            else:
                t = clamp(shm_throttle, 0.0, 1.0)
                b = clamp(shm_brake, 0.0, 1.0)
                s = clamp(shm_steering, -1.0, 1.0)
                
                apply_controls(t, b, s)
                
                latest["throttle"] = t
                latest["brake"] = b
                latest["steering"] = s
                latest["timestamp_ms"] = now

if __name__ == "__main__":
    print("[control_input_node] Running in headless mode", flush=True)
    threading.Thread(target=watchdog_and_control_loop, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            with client_lock:
                car_controls.throttle = 0.0
                car_controls.brake = 1.0
                car_controls.steering = 0.0
                client.setCarControls(car_controls, VEHICLE_NAME)
                client.enableApiControl(False, VEHICLE_NAME)
        except Exception:
            pass
        print("[control_input_node] Stopped", flush=True)