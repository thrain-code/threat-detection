# main.py
import cv2
import datetime
from ultralytics import YOLO
from config import ALERT_OBJECTS, CONFIDENCE_THRESHOLD, MIN_OBJECT_AREA

# Gunakan model yang lebih besar untuk akurasi lebih tinggi
model = YOLO("models/yolov8m.pt")  # Ganti dengan yolov8m.pt atau yolov8l.pt

# Konfigurasi VideoCapture untuk kamera IP (jika diperlukan)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Fungsi untuk menyimpan log deteksi
def save_log(label, x1, y1):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/security_log.txt", "a") as log_file:
        log_file.write(f"{timestamp}: {label} detected at ({x1}, {y1})\n")

# Kooldown alert untuk menghindari spam log
last_alert = {}
ALERT_COOLDOWN = 5  # detik

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tingkatkan kualitas frame dengan preprocessing
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Deteksi objek dengan parameter optimasi
    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.45,  # Lower IOU threshold untuk deteksi yang lebih ketat
        persist=True,  # Tracking yang persisten
        verbose=False,  # Non-aktifkan output console
        device='cpu'  # Gunakan GPU jika tersedia
    )

    # Proses hasil deteksi
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            object_id = int(box.id[0]) if box.id is not None else None

            # Filter berdasarkan ukuran objek
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_OBJECT_AREA:
                continue

            # Gambar bounding box
            color = (0, 255, 0) if label not in ALERT_OBJECTS else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f} ID:{object_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Sistem alert dengan kooldown
            if label in ALERT_OBJECTS and confidence > CONFIDENCE_THRESHOLD:
                current_time = datetime.datetime.now()
                if object_id not in last_alert or (current_time - last_alert[object_id]).seconds > ALERT_COOLDOWN:
                    print(f"⚠️ WARNING: {label} (ID:{object_id}) detected with confidence {confidence:.2f}!")
                    save_log(label, x1, y1)
                    last_alert[object_id] = current_time

    # Tampilkan video
    cv2.imshow("CCTV Security Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()