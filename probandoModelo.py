from ultralytics import YOLO

model = YOLO("outputs/train_run/weights/best.pt")
results = model(
    "videos/prueba2.mp4", 
    save=True, conf=0.2,
    project="outputs", 
    name="predict_run"
    )
