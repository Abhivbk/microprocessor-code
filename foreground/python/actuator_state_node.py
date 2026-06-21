import sys
import os
import time
import tkinter as tk
import mmap
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

VEHICLE_NAME = "FSCar"

SHARED_MEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground"))
ACTUATOR_BIN_PATH = os.path.join(SHARED_MEM_DIR, "abs_current.bin")

BINARY_FORMAT = "<Qfff"
STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

os.makedirs(SHARED_MEM_DIR, exist_ok=True)
if not os.path.exists(ACTUATOR_BIN_PATH) or os.path.getsize(ACTUATOR_BIN_PATH) != STRUCT_SIZE:
    with open(ACTUATOR_BIN_PATH, "wb") as f:
        f.write(b"\x00" * STRUCT_SIZE)

act_file = open(ACTUATOR_BIN_PATH, "r+b")
ram = mmap.mmap(act_file.fileno(), STRUCT_SIZE)

client = fsds.FSDSClient()
client.confirmConnection()

def read_controls():
    controls = client.getCarControls(VEHICLE_NAME)
    return {
        "timestamp_ms": int(time.time() * 1000),
        "throttle": float(controls.throttle),
        "brake": float(controls.brake),
        "steering": float(controls.steering),
    }

root = tk.Tk()
root.title("Actuator State Node")
root.geometry("500x220")
root.configure(bg="#1a1a1a")

title_label = tk.Label(root, text="Actuator State", font=("Arial", 18, "bold"), fg="cyan", bg="#1a1a1a")
title_label.pack(pady=10)

throttle_var = tk.StringVar(value="Throttle: 0.000")
brake_var = tk.StringVar(value="Brake: 0.000")
steering_var = tk.StringVar(value="Steering: 0.000")
status_var = tk.StringVar(value="Status: Running")

for var in [throttle_var, brake_var, steering_var, status_var]:
    tk.Label(root, textvariable=var, font=("Arial", 14), fg="white", bg="#1a1a1a", anchor="w").pack(fill="x", padx=20, pady=5)

def update_ui():
    try:
        data = read_controls()
        throttle_var.set(f"Throttle: {data['throttle']:.3f}")
        brake_var.set(f"Brake: {data['brake']:.3f}")
        steering_var.set(f"Steering: {data['steering']:.3f}")
        status_var.set("Status: OK")
        
        packed = struct.pack(
            BINARY_FORMAT,
            data["timestamp_ms"],
            data["throttle"],
            data["brake"],
            data["steering"]
        )
        ram[0:STRUCT_SIZE] = packed
        
    except Exception as e:
        throttle_var.set("Throttle: 0.000")
        brake_var.set("Brake: 0.000")
        steering_var.set("Steering: 0.000")
        status_var.set(f"Status: Error - {e}")

    root.after(50, update_ui)

def on_close():
    ram.close()
    act_file.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, update_ui)
root.mainloop()