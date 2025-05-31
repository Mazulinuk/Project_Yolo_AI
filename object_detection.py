import torch
from ultralytics import YOLO
import cv2
import numpy as np

from ultralytics import YOLO

#default_model = YOLO("yolov8m.pt")
#trained_model = YOLO("runs/detect/train/weights/best.pt")

#model = trained_model  # ← Або default_model, коли потрібно
#model = YOLO("D:/Project/runs/detect/train3/weights/best.pt")
model = YOLO("yolov8m.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def detect_objects(img):

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    height, width, _ = img.shape
    resized_img = cv2.resize(img, (1280, 1280))

    results = model(resized_img, device=device)

    detected_objects = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()

            if score > 0.5:
                x1 = int(x1 * (width / 1280))
                y1 = int(y1 * (height / 1280))
                x2 = int(x2 * (width / 1280))
                y2 = int(y2 * (height / 1280))

                label = model.names[int(class_id)]

               #if label == "person":
                #    label = "car"
               # elif label == "car":
                #    label = "person"
            # detected_objects.append((x1, y1, x2, y2, label))

    return detected_objects
import torch
from ultralytics import YOLO
import cv2

#default_model = YOLO("yolov8m.pt")
#trained_model = YOLO("runs/detect/train/weights/best.pt")

#model = trained_model
#model = YOLO("D:/Project/runs/detect/train3/weights/best.pt")
model = YOLO("yolov8m.pt")

#model = YOLO("D:/Project/runs/detect/train3/weights/best.pt") #18 #3
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

CLASS_MAP = {
    "person": "person",
    "civilian": "person",
    "soldier": "person",
    "camouflage_soldier": "person",

    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bike": "vehicle",
    "civilian_vehicle": "vehicle",
    "military_vehicle": "vehicle",
    "armored_vehicle": "vehicle",
    "tank": "vehicle",
    "artillery": "vehicle",

    "aircraft": "aircraft",
    "drone": "aircraft",
    "military_aircraft": "aircraft",
    "military_warship": "aircraft",
}

def detect_objects(img):

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    height, width, _ = img.shape
    resized_img = cv2.resize(img, (1280, 1280))

    results = model(resized_img, device=device)

    detected_objects = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()

            if score > 0.5:
                x1 = int(x1 * (width / 1280))
                y1 = int(y1 * (height / 1280))
                x2 = int(x2 * (width / 1280))
                y2 = int(y2 * (height / 1280))

                orig_label = model.names[int(class_id)]
                label = CLASS_MAP.get(orig_label, "unknown")
                import torch
from ultralytics import YOLO
import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolov8m.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def detect_objects(img):

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    height, width, _ = img.shape
    resized_img = cv2.resize(img, (1280, 1280))
    results = model(resized_img, device=device)

    detected_objects = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()

            if score > 0.5:
                x1 = int(x1 * (width / 1280))
                y1 = int(y1 * (height / 1280))
                x2 = int(x2 * (width / 1280))
                y2 = int(y2 * (height / 1280))

                label = model.names[int(class_id)]

    return detected_objects
import torch
from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")
#model = YOLO("D:/Project/runs/detect/train3/weights/best.pt") #18 #3
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

CLASS_MAP = {
    "person": "person",
    "civilian": "person",
    "soldier": "person",
    "camouflage_soldier": "person",

    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bike": "vehicle",
    "civilian_vehicle": "vehicle",
    "military_vehicle": "vehicle",
    "armored_vehicle": "vehicle",
    "tank": "vehicle",
    "artillery": "vehicle",

    "aircraft": "aircraft",
    "drone": "aircraft",
    "military_aircraft": "aircraft",
    "military_warship": "aircraft",
}

def detect_objects(img):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    height, width, _ = img.shape
    resized_img = cv2.resize(img, (1280, 1280))
    results = model(resized_img, device=device)
    detected_objects = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()

            if score > 0.5:
                x1 = int(x1 * (width / 1280))
                y1 = int(y1 * (height / 1280))
                x2 = int(x2 * (width / 1280))
                y2 = int(y2 * (height / 1280))
                orig_label = model.names[int(class_id)]
                label = CLASS_MAP.get(orig_label, "unknown")
                detected_objects.append((x1, y1, x2, y2, label))

    return detected_objects



