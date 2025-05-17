import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# Load the model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", 
                        (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def run_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(frame)
        cv2.imshow("Webcam - Real-Time Object Detection", frame)
        if cv2.waitKey(1) == ord("q") or cv2.getWindowProperty("Webcam - Real-Time Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image_detection(img_path):
    image = cv2.imread(img_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image!")
        return
    output = detect_objects(image)
    cv2.imshow("Image - Object Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def start_webcam_thread():
    threading.Thread(target=run_webcam, daemon=True).start()

def select_image_and_detect():
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if img_path:
        threading.Thread(target=run_image_detection, args=(img_path,), daemon=True).start()

# GUI setup
root = tk.Tk()
root.title("Object Detection with MobileNet SSD")

label = tk.Label(root, text="Choose Detection Mode", font=("Arial", 14))
label.pack(pady=10)

btn_webcam = tk.Button(root, text="Webcam Detection", width=25, command=start_webcam_thread)
btn_webcam.pack(pady=5)

btn_image = tk.Button(root, text="Image File Detection", width=25, command=select_image_and_detect)
btn_image.pack(pady=5)

btn_quit = tk.Button(root, text="Quit", width=25, command=root.destroy)
btn_quit.pack(pady=20)

root.mainloop()
