import os
import time
import mmap
import struct
import numpy as np
from ultralytics import YOLO

SHARED_MEM_DIR_FG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sharedmemory", "forground"))
SHARED_MEM_DIR_BG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sharedmemory", "background"))
CAM_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "cam.bin")
CAM_OUT_PATH = os.path.join(SHARED_MEM_DIR_BG, "camera_cones.bin")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best.pt")

try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"[camera_cone_detection] YOLO loaded from {MODEL_PATH}")
except Exception as e:
    yolo_model = None
    print(f"[camera_cone_detection] failed to load model: {e}")

MAX_CONES = 100
OUT_SIZE = 8 + MAX_CONES * 24

os.makedirs(SHARED_MEM_DIR_BG, exist_ok=True)
if not os.path.exists(CAM_OUT_PATH) or os.path.getsize(CAM_OUT_PATH) != OUT_SIZE:
    with open(CAM_OUT_PATH, "wb") as f:
        f.write(b"\x00" * OUT_SIZE)

out_file = open(CAM_OUT_PATH, "r+b")
out_ram = mmap.mmap(out_file.fileno(), OUT_SIZE)

LABEL_MAP = {"yellow": 0, "blue": 1, "orange": 2}

def process_camera():
    if yolo_model is None or not os.path.exists(CAM_BIN_PATH):
        return

    try:
        with open(CAM_BIN_PATH, "r+b") as f:
            file_size = os.fstat(f.fileno()).st_size
            if file_size < 32:
                return
            
            ram = mmap.mmap(f.fileno(), file_size, access=mmap.ACCESS_READ)
            
            height, width, channels, dtype_code = struct.unpack("QQQQ", ram[0:32])
            img_size = height * width * channels
            
            if file_size < 32 + img_size:
                ram.close()
                return

            raw_bytes = ram[32:32+img_size]
            cam_img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, channels))
            ram.close()
            
    except Exception as e:
        return

    try:
        results = yolo_model(cam_img, verbose=False)[0]

        raw_cam_cones = []
        CAM_W = width
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results.names.get(cls_id, str(cls_id))
            
            if conf < 0.60:
                continue
                
            if x1 <= 5 or x2 >= CAM_W - 5:
                continue
                
            bcx = (x1 + x2) / 2
            raw_cam_cones.append((bcx, x1, y1, x2, y2, label, conf))

        cam_cones = []
        raw_cam_cones.sort(key=lambda c: c[6], reverse=True)
        for new_cone in raw_cam_cones:
            _, x1_n, y1_n, x2_n, y2_n, _, _ = new_cone
            overlap = False
            for kept_cone in cam_cones:
                _, x1_k, y1_k, x2_k, y2_k, _, _ = kept_cone
                xx1 = max(x1_n, x1_k)
                yy1 = max(y1_n, y1_k)
                xx2 = min(x2_n, x2_k)
                yy2 = min(y2_n, y2_k)
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area_n = (x2_n - x1_n) * (y2_n - y1_n)
                area_k = (x2_k - x1_k) * (y2_k - y1_k)
                union = float(area_n + area_k - inter)
                if union > 0:
                    iou = inter / union
                else:
                    iou = 0.0
                if iou > 0.3:
                    overlap = True
                    break
            if not overlap:
                cam_cones.append(new_cone)

        cam_cones.sort(key=lambda c: c[4], reverse=True)
        
        num_cones = min(len(cam_cones), MAX_CONES)
        out_ram[0:8] = struct.pack("Q", num_cones)
        
        offset = 8
        for i in range(num_cones):
            c = cam_cones[i]
            bcx, x1, y1, x2, y2, label_str, conf = c
            label_id = 0
            for k, v in LABEL_MAP.items():
                if k in label_str.lower():
                    label_id = v
                    break
            
            out_ram[offset:offset+24] = struct.pack("fffffi", float(x1), float(y1), float(x2), float(y2), float(conf), label_id)
            offset += 24

    except Exception as e:
        print(f"[camera_cone_detection] error: {e}")

if __name__ == "__main__":
    print("[camera_cone_detection] Running...", flush=True)
    try:
        while True:
            process_camera()
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("[camera_cone_detection] Stopped.")
        out_ram.close()
        out_file.close()
