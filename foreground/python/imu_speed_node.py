import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
IMU_NAME = "Imu"
PORT = 81


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print("[imu_speed_node] Connected to FSDS")
            return client
        except Exception as e:
            print(f"[imu_speed_node] FSDS not ready: {e}")
            time.sleep(2)


client = connect_fsds_forever()

tcp = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT)
tcp.start()


def get_packet():
    global client
    try:
        state = client.getCarState(VEHICLE_NAME)
        imu = client.getImuData(imu_name=IMU_NAME, vehicle_name=VEHICLE_NAME)
    except Exception:
        client = connect_fsds_forever()
        state = client.getCarState(VEHICLE_NAME)
        imu = client.getImuData(imu_name=IMU_NAME, vehicle_name=VEHICLE_NAME)

    return {
        "timestamp_ms": int(time.time() * 1000),
        "ground_speed_mps": float(state.speed),
        "imu": {
            "angular_velocity": {
                "x": float(imu.angular_velocity.x_val),
                "y": float(imu.angular_velocity.y_val),
                "z": float(imu.angular_velocity.z_val),
            },
            "linear_acceleration": {
                "x": float(imu.linear_acceleration.x_val),
                "y": float(imu.linear_acceleration.y_val),
                "z": float(imu.linear_acceleration.z_val),
            },
            "orientation": {
                "x": float(imu.orientation.x_val),
                "y": float(imu.orientation.y_val),
                "z": float(imu.orientation.z_val),
                "w": float(imu.orientation.w_val),
            },
        },
    }


if __name__ == "__main__":
    print("[imu_speed_node] Running in headless mode", flush=True)
    try:
        while True:
            start_time = time.time()
            try:
                data = get_packet()
                tcp.send_json(data)
            except Exception as e:
                print(f"[imu_speed_node] Error: {e}", flush=True)

            # Maintain ~20 Hz loop rate (approx 50ms)
            elapsed = time.time() - start_time
            time.sleep(max(0.005, 0.05 - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        tcp.stop()
        print("[imu_speed_node] Stopped", flush=True)