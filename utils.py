import cv2
import time
import csv
import os
import numpy as np

try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    FACE_CASCADE = None


def setup_logging(log_file, screenshot_dir):
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "duration_seconds", "screenshot_path"])


def blur_faces(image):
    if FACE_CASCADE is None or FACE_CASCADE.empty():
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        face_roi = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (101, 101), 0)
        image[y1:y2, x1:x2] = blurred_face

    return image


def log_event(frame, label, duration_seconds, log_file, screenshot_dir):
    blurred_frame = blur_faces(frame.copy())

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp_str}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, blurred_frame)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_str, label, round(duration_seconds, 2), filepath])

    print(f"--- EVENT LOGGED: {label} (Duration: {round(duration_seconds, 2)}s) ---")


def frames_to_time_str(frames, fps):
    if fps == 0: return "00:00:00"
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"