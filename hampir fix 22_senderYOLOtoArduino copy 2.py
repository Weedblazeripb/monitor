import cv2
import torch
import numpy as np
import pathlib
from pathlib import Path
import sys
import serial
import time

# Menambahkan jalur direktori YOLOv5 ke sys.path
sys.path.append(str(Path(__file__).resolve().parents[0]))
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Establish serial connection with Arduino
ser = serial.Serial('COM9', 9600)  # Adjust 'COM6' to your Arduino's port

def send_to_arduino(data):
    ser.write(data.encode())
    time.sleep(2)  # Wait 2 seconds to ensure Arduino receives and processes data

# Load model
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend("best.pt", device=device)

# Initialize camera
cap = cv2.VideoCapture(0)

# Define conversion factors from pixels to millimeters
x_conversion_factor = 720 / 640
y_conversion_factor = 610 / 480

# Inference Loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
                        if y_center >= 240:
                            Sx = int(-((x_center - 320) * x_conversion_factor))
                            Sy = int(-((y_center - 240) * y_conversion_factor))
                            if Sx and Sy:
                                inputX = '1'
                        if y_center < 240:
                            Sx = int(((x_center - 320) * x_conversion_factor))
                            Sy = int(((y_center - 240) * y_conversion_factor))
                            if Sx and Sy:
                                inputX = '1'
                    if x_center <= 320:
                        if y_center >= 240:
                            Sx = int(-((320 - x_center) * x_conversion_factor))
                            Sy = int(-((y_center - 240) * y_conversion_factor))
                            if Sx and Sy:
                                inputX = '1'
                        if y_center < 240:
                            Sx = int(((320 - x_center) * x_conversion_factor))
                            Sy = int(((240 - y_center) * y_conversion_factor))
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

        # Display the resulting frame outside the object detection loop
        cv2.imshow('YOLOv5', im0)
        # Exit condition when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt  # Raise KeyboardInterrupt to exit the loop

except KeyboardInterrupt:
    pass  # Handle KeyboardInterrupt by doing nothing

# Release resources
ser.close()
cap.release()
cv2.destroyAllWindows()
