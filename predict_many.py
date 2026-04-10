from ultralytics import YOLO
import os

model = YOLO("best.pt")

folder = "test/severe"   # change this to any class folder

for image_name in os.listdir(folder):
    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder, image_name)
        results = model(image_path)

        top1_index = results[0].probs.top1
        top1_conf = results[0].probs.top1conf.item()
        class_name = results[0].names[top1_index]

        print(f"Image: {image_name}")
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {round(top1_conf * 100, 2)}%")
        print("-" * 30)