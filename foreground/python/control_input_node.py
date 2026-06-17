import sys
import os
import time
import socket
import json
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
HOST = "0.0.0.0"
PORT_CONTROL_IN = 82
PORT_ACTUATOR_OUT = 83


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            client.enableApiControl(True, VEHICLE_NAME)
            print("[control_input_node] Connected to FSDS")
            return client
        except Exception as e:
            print(f"[control_input_node] FSDS not ready: {e}")
            time.sleep(2)


client = connect_fsds_forever()
car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 1.0
car_controls.steering = 0.0

client_lock = threading.Lock()

tcp_pub = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT_ACTUATOR_OUT)
tcp_pub.start()

latest = {
    "throttle": 0.0,
    "brake": 1.0,
    "steering": 0.0,
    "status": "Listening",
    "timestamp_ms": int(time.time() * 1000),
}

desired_controls = {
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
            latest["status"] = "Reconnecting FSDS..."
            client = connect_fsds_forever()
            try:
                client.setCarControls(car_controls, VEHICLE_NAME)
            except Exception:
                pass


def publish_latest():
    payload = {
        "timestamp_ms": latest["timestamp_ms"],
        "throttle": latest["throttle"],
        "brake": latest["brake"],
        "steering": latest["steering"],
    }
    tcp_pub.send_json(payload)


def handle_client(conn, addr):
    print(f"[control_input_node] Client connected: {addr}", flush=True)
    latest["status"] = f"Connected: {addr}"

    try:
        file = conn.makefile("r")
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except Exception as e:
                print(f"[control_input_node] Bad JSON: {e} | line={line!r}")
                continue

            throttle = clamp(float(msg.get("throttle", 0.0)), 0.0, 1.0)
            brake = clamp(float(msg.get("brake", 0.0)), 0.0, 1.0)
            steering = clamp(float(msg.get("steering", 0.0)), -1.0, 1.0)

            desired_controls["throttle"] = throttle
            desired_controls["brake"] = brake
            desired_controls["steering"] = steering
            desired_controls["timestamp_ms"] = int(time.time() * 1000)

    except Exception as e:
        print(f"[control_input_node] Client handler error: {e}")
        latest["status"] = f"Client error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[control_input_node] Client disconnected: {addr}")


def watchdog_and_control_loop():
    print("[control_input_node] Control loop started", flush=True)
    while True:
        time.sleep(0.05)
        now = int(time.time() * 1000)
        last_time = desired_controls["timestamp_ms"]
        
        is_moving = (latest["throttle"] > 0.0 or latest["brake"] < 1.0)
        
        if now - last_time > 300:
            if is_moving:
                print("[control_input_node] WATCHDOG TRIGGERED: Active Braking due to packet timeout.", flush=True)
                apply_controls(0.0, 1.0, 0.0)
                latest["throttle"] = 0.0
                latest["brake"] = 1.0
                latest["steering"] = 0.0
                latest["timestamp_ms"] = now
                latest["status"] = "Watchdog active braking"
                publish_latest()
        else:
            t = desired_controls["throttle"]
            b = desired_controls["brake"]
            s = desired_controls["steering"]
            
            apply_controls(t, b, s)
            
            latest["throttle"] = t
            latest["brake"] = b
            latest["steering"] = s
            latest["timestamp_ms"] = now
            latest["status"] = "Command applied"
            publish_latest()


def server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT_CONTROL_IN))
    server.listen(5)

    print(f"[control_input_node] Listening on {HOST}:{PORT_CONTROL_IN}", flush=True)

    while True:
        try:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
        except Exception as e:
            print(f"[control_input_node] Accept error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    print("[control_input_node] Running in headless mode", flush=True)
    threading.Thread(target=server_thread, daemon=True).start()
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
        tcp_pub.stop()
        print("[control_input_node] Stopped", flush=True)