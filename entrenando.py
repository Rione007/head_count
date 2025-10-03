from ultralytics import YOLO

model = YOLO("yolov8s.pt")  

model.train(
    data="datasets2/data.yaml",
    epochs=50,
    imgsz=960,
    batch=8, #imagenes procesadas en pararlelo
    project="outputs",
    name="head_detector_v2"
)

