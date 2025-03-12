from ultralytics import YOLO
import cv2
import cvzone
import math

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

model = YOLO('YOLO Weights\\yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = capture.read()
    # result = model(img, show=True)
    result = model(img, stream = True)

    for i in result:
        boxes = i.boxes
        for box in boxes:
            ## Bouding Box
            ## for OpenCv
            x1, y1, x2, y2 = box.xyxy[0]        
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 255), 2)

            # For Cvzone
            # x1, y1, w, h = box.xyxy[0] 
            # bbox = int(x1), int(y1), int(w), int(h)
            # cvzone.cornerRect(img, bbox)
            
            ## Confidence
            conf = math.ceil(box.conf[0]*100) / 100

            ## Class
            cls = box.cls[0]
            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale = 2)


    cv2.imshow("Screen", img)
    cv2.waitKey(1)