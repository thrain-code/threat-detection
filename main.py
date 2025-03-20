# main.py
import cv2
import datetime
from ultralytics import YOLO
from config import ALERT_OBJECTS

# Load model YOLO
model = YOLO("models/yolov8n.pt")

# Buka video CCTV (ganti "0" dengan URL CCTV jika ada)
cap = cv2.VideoCapture(0)

# Fungsi untuk menyimpan log deteksi
def save_log(label, x1, y1):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/security_log.txt", "a") as log_file:
        log_file.write(f"{timestamp}: {label} detected at ({x1}, {y1})\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dalam frame
    results = model(frame)

    # Proses hasil deteksi
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Simpan log jika objek mencurigakan terdeteksi
            if label in ALERT_OBJECTS:
                print(f"⚠️ WARNING: {label} detected!")
                save_log(label, x1, y1)

    # Tampilkan video
    cv2.imshow("CCTV Security Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
