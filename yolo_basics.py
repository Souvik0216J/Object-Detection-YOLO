from ultralytics import YOLO
import cv2
model = YOLO('YOLO Weights\\yolov8n.pt')
result = model("assets\\3.jpg", show=True)

cv2.waitKey(0)
