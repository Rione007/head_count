import cv2
from ultralytics import YOLO

# Modelo YOLOv8 con tracking
model = YOLO("yolov8n.pt")

# Video
cap = cv2.VideoCapture("C:\\Users\\ander\\Music\\bus_counter\\videos\\prueba2.mp4")

# Leer primer frame
ret, frame = cap.read()
if not ret:
    print("No se pudo abrir el video")
    exit()

# Rotar
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
height, width, _ = frame.shape

# Línea de conteo
line_y = height // 2

entradas = 0
salidas = 0
tracks = {}  # track_id -> última posición (cx, cy)

# Reiniciar video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotar frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Tracking
    results = model.track(frame, persist=True, verbose=False)

    # Dibujar línea
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes, results[0].boxes.id.int().cpu().tolist()):
            cls = int(box.cls[0])
            if cls == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Dibujar bbox y centro
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Si ya teníamos posición previa, dibujar flecha
                if track_id in tracks:
                    prev_cx, prev_cy = tracks[track_id]

                    # Dibujar flecha movimiento
                    cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (255, 0, 0), 2, tipLength=0.5)

                    # Detectar cruce
                    if prev_cy < line_y <= cy:   # cruzó hacia abajo → salida
                        salidas += 1
                        tracks.pop(track_id)  # evitar dobles conteos
                    elif prev_cy > line_y >= cy:  # cruzó hacia arriba → entrada
                        entradas += 1
                        tracks.pop(track_id)

                # Actualizar posición
                tracks[track_id] = (cx, cy)

    # Mostrar contadores
    cv2.putText(frame, f"Entradas: {entradas}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Salidas: {salidas}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Bus Counter", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
