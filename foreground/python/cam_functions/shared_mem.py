import mmap
import struct
from pathlib import Path
import numpy as np
import cv2
import numpy as np

CAM_FILE_PATH = Path(__file__).resolve().parents[3] / "sharedmemory" / "forground" / "cam.bin"

def save_to_shared_memory(frame, filename=CAM_FILE_PATH):
    """
    Save camera frame / cam_panel to a memory-mapped binary file.

    Format:
    bytes 0-31   : header
        uint64 height
        uint64 width
        uint64 channels
        uint64 dtype_code  # 1 = uint8
    bytes 32-end : raw image bytes
    """

    if frame is None:
        return

    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if frame.ndim != 3:
        raise ValueError(f"Expected frame shape (H, W, C), got {frame.shape}")

    height, width, channels = frame.shape

    HEADER_SIZE = 32
    DTYPE_CODE_UINT8 = 1
    image_size = height * width * channels
    total_size = HEADER_SIZE + image_size

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_bytes = frame.tobytes()

    import time
    for attempt in range(25):
        try:
            with open(out_path, "wb") as f:
                f.write(b"\x00" * total_size)

            with open(out_path, "r+b") as f:
                ram = mmap.mmap(f.fileno(), total_size)

                ram[0:32] = struct.pack(
                    "QQQQ",
                    height,
                    width,
                    channels,
                    DTYPE_CODE_UINT8
                )

                ram[HEADER_SIZE:HEADER_SIZE + image_size] = frame_bytes

                ram.flush()
                ram.close()
            break
        except (PermissionError, OSError) as e:
            if attempt == 24:
                print(f"[warning] Failed to write camera shared memory: {e}")
            time.sleep(0.005)



