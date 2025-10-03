import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=60, max_distance=100):
        self.next_object_id = 0
        self.objects = {}            # id -> (cx, cy)
        self.disappeared = {}
        self.prev_x = {}             # id -> previous x
        self.prev_y = {}             # id -> previous y
        self.counted = {}            # id -> counted flag
        # extra flags for two_lines / roi
        self.passed_line1 = {}       # for two_lines: whether passed first line
        self.in_roi = {}             # for roi: whether currently inside roi
        self.entry_side = {}         # for roi: where entered from ('left'/'right')
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.prev_x[oid] = centroid[0]
        self.prev_y[oid] = centroid[1]
        self.counted[oid] = 0
        self.passed_line1[oid] = False
        self.in_roi[oid] = False
        self.entry_side[oid] = None
        self.next_object_id += 1

    def deregister(self, object_id):
        for d in (self.objects, self.disappeared, self.prev_x, self.prev_y,
                  self.counted, self.passed_line1, self.in_roi, self.entry_side):
            if object_id in d:
                del d[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (r, c) in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            self.objects[oid] = tuple(input_centroids[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for r in unused_rows:
            oid = object_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for c in unused_cols:
            self.register(tuple(input_centroids[c]))

        return self.objects

# ---------------- MAIN ----------------
# Elige modo: "vertical", "two_lines_vertical", "roi_vertical"
MODE = "vertical"

# Detector
model = YOLO("yolov8n.pt")

VIDEO_PATH = r"C:\Users\ander\Music\bus_counter\videos\prueba2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("No se pudo abrir el video")
    exit()

# Trabajamos en la orientación ORIGINAL (sin rotar).
height, width = frame.shape[:2]

# --- Parámetros verticales ---
# Modo vertical (una sola línea vertical)
line_x = width // 2
line_offset = 10   # tolerancia en px para evitar jitter

# Modo two_lines_vertical (dos líneas verticales)
line1_x = int(width * 0.45)
line2_x = int(width * 0.55)
two_lines_offset = 8

# Modo ROI vertical (rectángulo). Define (x1,y1) top-left y (x2,y2) bottom-right
# Para ROI vertical piénsatelo como una puerta vertical: si entra por left y sale por right -> cuenta
ROI = (int(width*0.35), int(height*0.25), int(width*0.65), int(height*0.75))

# Tracker
tracker = CentroidTracker(max_disappeared=90, max_distance=160)

entradas = 0   # left -> right
salidas = 0    # right -> left

# Detección
CONF_THRES = 0.20
IMG_SZ = 1280
MIN_AREA = 0  # 0 = aceptar todas las detecciones

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vis = frame.copy()
    results = model(frame, imgsz=IMG_SZ, conf=CONF_THRES, verbose=False)[0]

    centroids = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls = int(cls); conf = float(conf)
        if cls == 0 and conf >= CONF_THRES:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            w = x2 - x1; h = y2 - y1; area = w*h
            if area < MIN_AREA:
                pass
            cx = int(x1 + w/2); cy = int(y1 + h/2)
            centroids.append((cx, cy))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(vis, (cx,cy), 4, (0,0,255), -1)

    objects = tracker.update(centroids)

    # ---- LOGICA VERTICAL SEGUN MODO ----
    if MODE == "vertical":
        # dibuja linea vertical
        cv2.line(vis, (line_x, 0), (line_x, height), (0,255,255), 2)
        cv2.putText(vis, f"Line X={line_x}", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        for oid, centroid in list(objects.items()):
            cx, cy = int(centroid[0]), int(centroid[1])
            prevx = tracker.prev_x.get(oid, cx)
            flag = tracker.counted.get(oid, 0)

            # CRUCE izquierda -> derecha
            if prevx < line_x - line_offset and cx > line_x + line_offset:
                if flag != 1:
                    salidas += 1
                    tracker.counted[oid] = 1
                    print(f"CRUCE L->R ID{oid} prev_x={prevx} cur_x={cx} salidas={salidas}")

            # CRUCE derecha -> izquierda
            elif prevx > line_x + line_offset and cx < line_x - line_offset:
                if flag != -1:
                    entradas += 1
                    tracker.counted[oid] = -1
                    print(f"CRUCE R->L ID{oid} prev_x={prevx} cur_x={cx} entradas={entradas}")

            # reset bandera si se aleja mucho (permitir recontar si vuelve)
            if abs(cx - line_x) > 80:
                tracker.counted[oid] = 0

            tracker.prev_x[oid] = cx
            tracker.prev_y[oid] = cy
            cv2.putText(vis, f"ID{oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    elif MODE == "two_lines_vertical":
        # dibuja las dos lineas verticales
        cv2.line(vis, (line1_x, 0), (line1_x, height), (255,255,0), 2)
        cv2.line(vis, (line2_x, 0), (line2_x, height), (255,255,0), 2)
        cv2.putText(vis, f"L1={line1_x} L2={line2_x}", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        for oid, centroid in list(objects.items()):
            cx, cy = int(centroid[0]), int(centroid[1])
            prevx = tracker.prev_x.get(oid, cx)

            # Si no ha pasado todavía la primera línea, detecta el paso por L1
            if not tracker.passed_line1.get(oid, False):
                # paso L1 de left->right
                if prevx < line1_x - two_lines_offset and cx > line1_x + two_lines_offset:
                    tracker.passed_line1[oid] = True
                    print(f"ID{oid} pasó L1 (L->R)")
                # paso L1 de right->left
                if prevx > line1_x + two_lines_offset and cx < line1_x - two_lines_offset:
                    tracker.passed_line1[oid] = True
                    print(f"ID{oid} pasó L1 (R->L)")
            else:
                # si ya pasó L1, espera a que pase L2 en la misma dirección para contar
                if prevx < line2_x - two_lines_offset and cx > line2_x + two_lines_offset:
                    entradas += 1   # L1 -> L2 left->right
                    tracker.passed_line1[oid] = False
                    tracker.counted[oid] = 1
                    print(f"ID{oid} contado L1->L2 entradas={entradas}")
                elif prevx > line2_x + two_lines_offset and cx < line2_x - two_lines_offset:
                    salidas += 1    # L1 -> L2 right->left
                    tracker.passed_line1[oid] = False
                    tracker.counted[oid] = -1
                    print(f"ID{oid} contado L2->L1 salidas={salidas}")

            # reset si se aleja mucho
            if abs(cx - ((line1_x+line2_x)//2)) > 180:
                tracker.passed_line1[oid] = False

            tracker.prev_x[oid] = cx
            tracker.prev_y[oid] = cy
            cv2.putText(vis, f"ID{oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    elif MODE == "roi_vertical":
        x1, y1, x2, y2 = ROI
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(vis, "ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        for oid, centroid in list(objects.items()):
            cx, cy = int(centroid[0]), int(centroid[1])
            prevx = tracker.prev_x.get(oid, cx)
            prev_in = tracker.in_roi.get(oid, False)
            now_in = (x1 <= cx <= x2 and y1 <= cy <= y2)

            # si entra al ROI, guardamos desde qué lado vino (left/right)
            if not prev_in and now_in:
                tracker.in_roi[oid] = True
                tracker.entry_side[oid] = 'left' if prevx < x1 else 'right'
                print(f"ID{oid} ENTER ROI from {tracker.entry_side[oid]}")

            # si estaba dentro y sale: decidir dirección según lado de salida
            if prev_in and not now_in:
                # salida por la izquierda
                if cx < x1:
                    # entró por right y salió por left -> cuenta como right->left
                    if tracker.entry_side.get(oid) == 'right':
                        salidas += 1
                        print(f"ID{oid} ROI EXIT LEFT salidas={salidas}")
                # salida por la derecha
                elif cx > x2:
                    # entró por left y salió por right -> cuenta left->right
                    if tracker.entry_side.get(oid) == 'left':
                        entradas += 1
                        print(f"ID{oid} ROI EXIT RIGHT entradas={entradas}")

                tracker.in_roi[oid] = False
                tracker.entry_side[oid] = None

            tracker.prev_x[oid] = cx
            tracker.prev_y[oid] = cy
            cv2.putText(vis, f"ID{oid}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # info en pantalla
    cv2.putText(vis, f"Entradas (R->L): {entradas}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(vis, f"Salidas (L->R): {salidas}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Counter Vertical", vis)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    if key == ord('m'):
        print("Tracker objects:", tracker.objects.keys())

cap.release()
cv2.destroyAllWindows()
