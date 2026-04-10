from ultralytics import YOLO
import os

# If last.pt exists, resume training from checkpoint
# Otherwise start fresh from the pretrained model
if os.path.exists("last.pt"):
    model = YOLO("last.pt")
    model.train(resume=True)
else:
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=".",
        epochs=50,
        imgsz=224,
        batch=16
    )