from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:/Project/dataset/dataset.yaml",
    epochs=40,
    batch=16,
    imgsz=640,
    device="cuda",
    workers=4,
    project="D:/Project/runs",
    name="custom_yolo",
    exist_ok=True,
    save=True
)

model.export(format="onnx")
