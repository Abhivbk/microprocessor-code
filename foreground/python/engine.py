import os
import subprocess
import socket
import time

import fsds


RPC_PORT = 41451


def simulator_rpc_ready():
    """Return True only when the service on the port answers FSDS RPC ping."""
    try:
        with socket.create_connection(("127.0.0.1", RPC_PORT), timeout=0.5):
            pass
        return bool(fsds.FSDSClient(timeout_value=1).ping())
    except Exception:
        return False


def wait_for_simulator(timeout=120):
    """Wait until the simulator RPC API is ready."""
    start = time.time()

    while time.time() - start < timeout:
        if simulator_rpc_ready():
            return True
        time.sleep(0.5)

    return False


def fsds_process_running():
    """Detect the FSDS launcher or its Unreal Engine child on Windows."""
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        process_list = result.stdout.lower()
        return '"fsds.exe"' in process_list or '"blocks.exe"' in process_list
    except (OSError, subprocess.SubprocessError):
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

    if simulator_rpc_ready():
        print("FSDS RPC is already available. Attaching to existing simulator...", flush=True)
        print("FSDS_READY", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        proc = None
        if fsds_process_running():
            print("FSDS process is already starting. Waiting for its RPC API...", flush=True)
        else:
            print("No existing FSDS instance found. Starting simulator...", flush=True)
            try:
                proc = run_fsds_as_spectator_server()
            except Exception as e:
                print(f"ERROR starting FSDS: {e}", flush=True)
                sys.exit(1)

        if wait_for_simulator():
            print("FSDS_READY", flush=True)
        else:
            print("ERROR: FSDS process exists but its RPC API did not become ready.", flush=True)
            sys.exit(1)

        try:
            while proc is None or proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            if proc is not None and proc.poll() is None:
                proc.terminate()
