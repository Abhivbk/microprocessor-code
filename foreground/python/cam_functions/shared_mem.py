import mmap
import struct
from pathlib import Path
import numpy as np
CAM_FILE_PATH = Path(__file__).resolve().parents[3] / "sharedmemory" / "forground" / "cam.bin"
HEADER_SIZE = 48
_mapped_file = _mapped_view = None
_mapped_size = 0
_mapped_path = None


def _open_map(path, size):
    """Reuse one map; a zero sequence means the payload is being replaced."""
    global _mapped_file, _mapped_view, _mapped_size, _mapped_path
    if _mapped_view is not None and _mapped_size == size and _mapped_path == path:
        return _mapped_view
    if _mapped_view is not None:
        _mapped_view.close()
        _mapped_file.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.truncate(size)
    _mapped_file = open(path, "r+b")
    _mapped_view = mmap.mmap(_mapped_file.fileno(), size)
    _mapped_size = size
    _mapped_path = path
    return _mapped_view

def save_to_shared_memory(frame, timestamp_ns=0, sequence=0, filename=CAM_FILE_PATH):
    """
    Save camera frame / cam_panel to a memory-mapped binary file.

    Format:
    bytes 0-47   : header (shape, dtype, simulator timestamp, pair sequence)
        uint64 height
        uint64 width
        uint64 channels
        uint64 dtype_code  # 1 = uint8
        uint64 timestamp_ns
    bytes 48-end : raw image bytes
    """

    if frame is None:
        return

    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (H, W, C), got {frame.shape}")

    height, width, channels = frame.shape

    DTYPE_CODE_UINT8 = 1
    image_size = height * width * channels
    total_size = HEADER_SIZE + image_size

    frame_bytes = frame.tobytes()
    ram = _open_map(Path(filename), total_size)
    ram[:HEADER_SIZE] = struct.pack("QQQQQQ", height, width, channels, DTYPE_CODE_UINT8, 0, 0)
    ram[HEADER_SIZE:] = frame_bytes
    ram[:HEADER_SIZE] = struct.pack(
        "QQQQQQ", height, width, channels, DTYPE_CODE_UINT8,
        int(timestamp_ns), int(sequence),
    )
