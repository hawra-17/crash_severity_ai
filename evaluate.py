from ultralytics import YOLO

model = YOLO("best.pt")

results = model.val(data=".", split="test")

print("=" * 40)
print(f"Top-1 Accuracy: {results.top1 * 100:.2f}%")
print(f"Top-5 Accuracy: {results.top5 * 100:.2f}%")
print("=" * 40)
