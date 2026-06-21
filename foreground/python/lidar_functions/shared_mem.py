import os
import mmap
import struct
from pathlib import Path
file_path = Path(__file__).resolve().parents[3] / "sharedmemory" / "forground" / "lid.bin"

def save_to_shared_memory(points, filename=None):
    """Save a collection of (x, y) points to a binary file using memory‑mapped I/O.

    If *filename* is not provided, the default path defined by *file_path* is used.
    The function creates the target directory if it does not exist.
    """
    if filename:
        out_paths = [Path(filename)]
    else:
        out_paths = [
            file_path,
            file_path.parent / "lidar.bin"
        ]

    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    num_points = len(points)
    HEADER_SIZE = 8
    PAIR_SIZE = 16
    total_size = HEADER_SIZE + num_points * PAIR_SIZE

    import time
    for out_path in out_paths:
        for attempt in range(25):
            try:
                # Pre‑allocate file with zeroes
                with open(out_path, "wb") as f:
                    f.write(b"\x00" * total_size)

                # Memory‑map and write data
                with open(out_path, "r+b") as f:
                    ram = mmap.mmap(f.fileno(), total_size)
                    ram[0:8] = struct.pack("Q", num_points)
                    offset = HEADER_SIZE
                    for x, y in points:
                        ram[offset:offset+8] = struct.pack("d", x)
                        offset += 8
                        ram[offset:offset+8] = struct.pack("d", y)
                        offset += 8
                    ram.flush()
                    ram.close()
                break
            except (PermissionError, OSError) as e:
                if attempt == 24:
                    print(f"[warning] Failed to write lidar shared memory to {out_path}: {e}")
                time.sleep(0.005)

def save_to_txt(points, filename="pointcloudelidar.txt"):
    """
    Saves a collection of 2D coordinates (x, y) to a text file.
    Each line in the file will contain a coordinate pair formatted as 'x,y'.
    """
    try:
        # Resolve path relative to the directory of this file (lidar_functions/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, "w") as f:
            for pt in points:
                f.write(f"{pt[0]},{pt[1]}\n")
    except Exception as e:
        print(f"[shared_mem] Error saving points to {filename}: {e}")


