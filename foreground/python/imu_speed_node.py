import sys
import os
import time
import mmap
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

VEHICLE_NAME = "FSCar"

SHARED_MEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground"))
IMU_BIN_PATH = os.path.join(SHARED_MEM_DIR, "ekfin_imu_groundspeed_gyro.bin")

# Receipt time detects frozen SHM; simulator time aligns IMU, camera and LiDAR.
BINARY_FORMAT = "<QQ11f"
STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            return client
        except Exception as e:
            print(f"[imu_speed_node] FSDS not ready: {e}", flush=True)
            time.sleep(2)

def update_imu():
    os.makedirs(SHARED_MEM_DIR, exist_ok=True)
    
    # Pre-allocate
    if not os.path.exists(IMU_BIN_PATH) or os.path.getsize(IMU_BIN_PATH) != STRUCT_SIZE:
        with open(IMU_BIN_PATH, "wb") as f:
            f.write(b"\x00" * STRUCT_SIZE)

    with open(IMU_BIN_PATH, "r+b") as f:
        ram = mmap.mmap(f.fileno(), STRUCT_SIZE)

        client = connect_fsds_forever()
        print("[imu_speed_node] Started...", flush=True)

        while True:
            try:
                state = client.getCarState(VEHICLE_NAME)
                imu_data = client.getImuData(vehicle_name=VEHICLE_NAME)

                now_ms = int(time.time() * 1000)
                speed = float(state.kinematics_estimated.linear_velocity.get_length())
                
                ang_vel = imu_data.angular_velocity
                lin_acc = imu_data.linear_acceleration
                ori = imu_data.orientation

                packed_data = struct.pack(
                    BINARY_FORMAT,
                    now_ms,
                    int(imu_data.time_stamp or state.timestamp),
                    speed,
                    float(ang_vel.x_val), float(ang_vel.y_val), float(ang_vel.z_val),
                    float(lin_acc.x_val), float(lin_acc.y_val), float(lin_acc.z_val),
                    float(ori.x_val), float(ori.y_val), float(ori.z_val), float(ori.w_val)
                )

                ram[0:STRUCT_SIZE] = packed_data
                time.sleep(0.05)

            except Exception as e:
                print(f"[imu_speed_node] Error: {e}", flush=True)
                client = connect_fsds_forever()

if __name__ == "__main__":
    update_imu()
