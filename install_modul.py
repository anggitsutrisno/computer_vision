from ultralytics import YOLO

model = YOLO("yolo26s.pt")

print("Model berhasil dimuat")
print(model.names)