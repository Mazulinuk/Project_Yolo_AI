import cv2
import os

image_folder = "dataset/images/train"
label_folder = "dataset/labels/train"

def yolo_to_bbox(yolo_coords, img_width, img_height):
    x_center, y_center, width, height = yolo_coords
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2

for image_name in os.listdir(image_folder):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape
        label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1, y1, x2, y2 = yolo_to_bbox((x_center, y_center, width, height), img_width, img_height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
  
        cv2.imshow("Image with Annotations", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()