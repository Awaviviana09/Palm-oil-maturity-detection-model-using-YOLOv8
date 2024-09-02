import torch
import cv2
import numpy as np
import pathlib
import os
from ultralytics import YOLO

# Menyesuaikan pathlib agar sesuai dengan OS Windows
pathlib.PosixPath = pathlib.WindowsPath

# Load model YOLOv8 yang sudah dilatih
model = YOLO(r'D:\ZAHWA\Program\detection-oil palm\best.pt')

# Define class names sesuai dengan dataset kelapa sawit
class_names = ['masak', 'kurang masak', 'mentah', 'terlalu masak']

# Pilih sumber: webcam, file video, atau gambar
source_type = input("Enter 'webcam' to use webcam, 'mp4' to use a video file, or 'image' to use an image file: ")

if source_type == 'webcam':
    cap = cv2.VideoCapture(0)  # Akses webcam
elif source_type == 'mp4':
    # Buka file video
    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)
elif source_type == 'image':
    # Buka file gambar
    image_path = input("Enter the path to the image file: ")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image. Please check the path and try again.")
        exit()
else:
    print("Invalid source type. Please choose 'webcam', 'mp4', or 'image'.")
    exit()

# Buat direktori untuk screenshot jika belum ada
screenshot_dir = r'D:\ZAHWA\Program\detection-oil palm\ScreenShoot'
os.makedirs(screenshot_dir, exist_ok=True)

# Inisialisasi counter untuk screenshot
screenshot_counter = 0

# Threshold untuk confidence score
confidence_threshold = 0.5

def process_frame(frame):
    global screenshot_counter
    # Lakukan deteksi objek pada frame
    results = model(frame)[0]  # Mendapatkan hasil deteksi pada frame

    for result in results.boxes:  # Iterasi melalui setiap hasil deteksi
        confidence = result.conf[0].item()  # Ambil nilai akurasi
        if confidence < confidence_threshold:
            continue  # Lewati deteksi dengan confidence di bawah threshold

        bbox = result.xyxy[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = bbox
        class_idx = int(result.cls[0])
        class_name = class_names[class_idx]

        # Debug print untuk memverifikasi deteksi
        print(f"Detected class: {class_name}, Confidence: {confidence:.2f}")

        # Set warna berdasarkan nama kelas
        if class_name == 'masak':
            color = (0, 0, 255)  # Warna merah untuk buah masak
        elif class_name == 'kurang masak':
            color = (255, 255, 0)  # Warna kuning untuk buah kurang masak
        elif class_name == 'terlalu masak':
            color = (0, 165, 255)  # Warna oranye untuk buah terlalu masak
        else:
            color = (128, 0, 128)  # Warna ungu untuk buah mentah

        # Gambar bounding box yang lebih tipis
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Label dengan font yang lebih rapi dan bayangan
        label = f"{class_name} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x = xmin
        text_y = ymin - 10 if ymin - 10 > 10 else ymin + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0] + 2, text_y + 2), color, -1)
        cv2.putText(frame, label, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Simpan screenshot jika mendeteksi buah masak
        if class_name == 'masak':
            # Crop area yang terdeteksi
            detected_area = frame[ymin:ymax, xmin:xmax]
            screenshot_path = os.path.join(screenshot_dir, f'screenshot_{class_name}_{screenshot_counter}.png')
            cv2.imwrite(screenshot_path, detected_area)
            screenshot_counter += 1

    return frame

if source_type in ['webcam', 'mp4']:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        # Tampilkan frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
else:
    frame = process_frame(frame)
    # Tampilkan frame
    cv2.imshow("frame", frame)
    cv2.waitKey(0)  # Tunggu sampai pengguna menekan sembarang tombol

cv2.destroyAllWindows()