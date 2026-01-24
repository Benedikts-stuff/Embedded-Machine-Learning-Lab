import cv2
import torch
import time
import os
import numpy as np
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=224, display_height=224, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "StateDicts", "pruned_checkpoint_30pct.pt")

checkpoint = torch.load(model_path, map_location=device)
model = TinyYoloV2FusedDynamic(num_classes=1, channels=checkpoint['cfg']).to(device)
model.load_state_dict(checkpoint['sd'])
model.eval()

res = 224 

print(f"Starte High-Speed Demo ({res}x{res})...")
cap = cv2.VideoCapture(gstreamer_pipeline(display_width=res, display_height=res), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden!")
    exit()

now = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor, yolo=True)

        detections = output.view(-1, 6)
        for box in detections[detections[:, 4] > 0.3]:
            cx, cy, w, h, conf, _ = box.tolist()
            
            x1, y1 = int((cx-w/2)*res), int((cy-h/2)*res)
            x2, y2 = int((cx+w/2)*res), int((cy+h/2)*res)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"P: {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        dt = time.time() - now
        now = time.time()
        fps_val = int(1/dt) if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps_val} (Res: {res})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Jetson High Speed Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
