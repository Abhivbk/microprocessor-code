import os
import subprocess
import socket
import time


RPC_PORT = 41451


def wait_for_port(port, timeout=120):
    """Wait until the simulator RPC port becomes available."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)

    return False


def run_fsds_as_spectator_server():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    exe_path = os.path.normpath(
        os.path.join(base_dir, "..", "engine_binaries", "FSDS.exe")
    )

    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"FSDS.exe not found at: {exe_path}")

    args = [exe_path, "/Game/TrainingMap?listen"]

    process = subprocess.Popen(
        args,
        cwd=os.path.dirname(exe_path)
    )

    return process


if __name__ == "__main__":
    import sys

    # Retry detection a few times — FSDS port may open slowly
    already_running = False
    for _ in range(5):
        try:
            with socket.create_connection(("127.0.0.1", RPC_PORT), timeout=2):
                already_running = True
                break
        except OSError:
            time.sleep(1)

    if already_running:
        print("FSDS is already running. Connecting to existing instance...", flush=True)
        print("FSDS_READY", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("Starting FSDS...", flush=True)
        try:
            proc = run_fsds_as_spectator_server()
        except Exception as e:
            print(f"ERROR starting FSDS: {e}", flush=True)
            sys.exit(1)

        print("Waiting for simulator RPC port...", flush=True)

        if wait_for_port(RPC_PORT):
            print("FSDS_READY", flush=True)
        else:
            # Port timed out but FSDS may still be loading — keep alive anyway
            print("WARNING: RPC port timeout — FSDS may still be loading. Staying alive...", flush=True)

        # Keep engine.py alive so Rust thread does not exit
        try:
            while proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            proc.terminate()