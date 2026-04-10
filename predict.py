from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test/minor/0128.JPEG")

top1_index = results[0].probs.top1
top1_conf = results[0].probs.top1conf.item()
class_name = results[0].names[top1_index]

print("Predicted class:", class_name)
print("Confidence:", round(top1_conf * 100, 2), "%")