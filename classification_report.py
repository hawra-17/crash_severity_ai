import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

model = YOLO("best.pt")

test_dir = "test"
class_names = sorted(os.listdir(test_dir))
class_names = [c for c in class_names if os.path.isdir(os.path.join(test_dir, c))]

y_true = []
y_pred = []

for class_name in class_names:
    class_folder = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_folder):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(class_folder, image_name)
            results = model(image_path, verbose=False)
            pred_index = results[0].probs.top1
            pred_class = results[0].names[pred_index]

            y_true.append(class_name)
            y_pred.append(pred_class)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, digits=4))

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
labels = sorted(set(y_true))
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(f"{'':12}" + "".join(f"{l:>12}" for l in labels))
for i, label in enumerate(labels):
    print(f"{label:12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels))))
