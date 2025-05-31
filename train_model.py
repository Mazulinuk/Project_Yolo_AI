from ultralytics import YOLO

def train_yolov8():
    model = YOLO("yolov8x.pt")
    
    # Налаштування тренування:
    # data
    # epochs
    # imgsz
    # batch
    # device
    # project
    # name
    
    model.train(
        data="D:/PROJECT/DATASET/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project="D:/PROJECT/YOLOv8_training",
        name="visdrone_yolov8x"
    )

if __name__ == "__main__":
    train_yolov8()
