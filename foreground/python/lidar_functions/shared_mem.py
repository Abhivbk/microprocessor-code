import mmap
import struct
from pathlib import Path

import numpy as np


FILE_PATH = Path(__file__).resolve().parents[3] / "sharedmemory" / "forground" / "lid.bin"
HEADER_SIZE = 24
_mapped_file = _mapped_view = None
_mapped_capacity = 0
_mapped_path = None


def _open_map(path, required_points):
    """Grow geometrically, then reuse the map for later scans."""
    global _mapped_file, _mapped_view, _mapped_capacity, _mapped_path
    if _mapped_view is not None and required_points <= _mapped_capacity and _mapped_path == path:
        return _mapped_view
    if _mapped_view is not None:
        _mapped_view.close()
        _mapped_file.close()
    _mapped_capacity = max(64, 1 << max(0, required_points - 1).bit_length())
    size = HEADER_SIZE + _mapped_capacity * 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.truncate(size)
    _mapped_file = open(path, "r+b")
    _mapped_view = mmap.mmap(_mapped_file.fileno(), size)
    _mapped_path = path
    return _mapped_view


def save_to_shared_memory(points, timestamp_ns=0, sequence=0, filename=None):
    """Publish one LiDAR scan without recreating the file each frame."""
    path = Path(filename) if filename else FILE_PATH
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(array)
    ram = _open_map(path, count)
    ram[:HEADER_SIZE] = struct.pack("QQQ", 0, 0, 0)
    if count:
        ram[HEADER_SIZE:HEADER_SIZE + count * 16] = array.tobytes()
    ram[:HEADER_SIZE] = struct.pack("QQQ", count, int(timestamp_ns), int(sequence))


def save_to_txt(points, filename="pointcloudelidar.txt"):
    """Write optional human-readable LiDAR coordinates for debugging."""
    path = Path(__file__).resolve().parent / filename
    try:
        with open(path, "w") as stream:
            for x, y in points:
                stream.write(f"{x},{y}\n")
    except OSError as error:
        print(f"[shared_mem] Error saving points to {filename}: {error}")
