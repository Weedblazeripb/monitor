from flask import Flask, render_template, Response, request
import cv2
import torch
import numpy as np
import pathlib
from pathlib import Path
import sys
import serial
import time
import threading

app = Flask(__name__)

# Menambahkan jalur direktori YOLOv5 ke sys.path
sys.path.append(str(Path(__file__).resolve().parents[0]))
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Establish serial connection with Arduino
ser = serial.Serial('COM6', 9600)  # Adjust 'COM6' to your Arduino's port

def send_to_arduino(data):
    ser.write(data.encode())
    time.sleep(2)  # Wait 2 seconds to ensure Arduino receives and processes data

# Load model
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend("C:/Users/Nibroos/OneDrive/Documents/pkm 2024/yolov5 pkmneww/yolov5 pkm/yolov5-master/best.pt", device=device)

# Initialize camera
cap = cv2.VideoCapture(0)

# Define conversion factors from pixels to millimeters
x_conversion_factor = 720 / 640
y_conversion_factor = 610 / 480

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Preprocess the frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(device)
            img = img.permute(2, 0, 1).float()  # HWC to CHW
            img /= 255.0  # Normalize to [0, 1]
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            
            # Process detections
            for det in pred:  # detections per image
                im0 = frame.copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

                        # Calculate centroid
                        x_center = int((xyxy[0] + xyxy[2]) / 2)
                        y_center = int((xyxy[1] + xyxy[3]) / 2)
                        Sx = 0
                        Sy = 0
                        inputX = '0'

                        if x_center > 320:
                            Sx = int((x_center) * x_conversion_factor)
                            if y_center > 240:
                                Sy = int((y_center) * y_conversion_factor)
                            else:
                                Sy = int(-(y_center) * y_conversion_factor)
                        else:
                            Sx = int(-(x_center) * x_conversion_factor)
                            if y_center > 240:
                                Sy = int((y_center) * y_conversion_factor)
                            else:
                                Sy = int(-(y_center) * y_conversion_factor)
                                
                        if Sx and Sy:
                            inputX = '1'

                        # Display centroid coordinates
                        centroid_text = f'({x_center}, {y_center})'
                        cv2.putText(im0, centroid_text, (x_center, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Draw centroid
                        cv2.circle(im0, (x_center, y_center), 5, (0, 255, 0), -1)
                        
                        # Send data to Arduino
                        send_to_arduino(f"{Sx} {Sy} {inputX}\n")

                        # Wait for 17 seconds before sending the next command (15 seconds for laser and 2 seconds buffer)
                        time.sleep(17)

                im0 = annotator.result()

            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_operation():
    # Logic to start the YOLO operation
    # This can be handled as needed
    return "Operation started"

@app.route('/stop', methods=['POST'])
def stop_operation():
    # Logic to stop the YOLO operation
    # This can be handled as needed
    return "Operation stopped"

if __name__ == "__main__":
    app.run(debug=True)
