import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak bisa dibuka")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

colors = {
    "Merah": ([0, 120, 70], [10, 255, 255]),
    "Kuning": ([20, 100, 100], [30, 255, 255]),
    "Hijau": ([36, 100, 100], [86, 255, 255]),
    "Biru": ([94, 80, 2], [126, 255, 255]),
    "Ungu": ([129, 50, 70], [158, 255, 255]),
    "Oranye": ([10, 100, 20], [25, 255, 255]),
    "Putih": ([0, 0, 200], [180, 50, 255]),
    "Hitam": ([0, 0, 0], [180, 255, 50])
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat menangkap gambar")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = []

    for color_name, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                detected_colors.append(color_name)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(thresh) > 3000:
        cv2.putText(frame, "Gerakan Terdeteksi!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    prev_gray = gray.copy()

    cv2.imshow("Deteksi Warna & Gerak", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
