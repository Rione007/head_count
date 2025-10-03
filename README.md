El dataset a utilizar sera el datasets2
corregir las ubicaciones del archivo data.yaml
 
train: C:\Users\ander\Music\bus_counter\datasets2\train\images    <- cambiar a las propias
val: C:\Users\ander\Music\bus_counter\datasets2\valid\images     <- cambiar a las propias

entrenando.py:
aqui se entrenara 
epoch : numero de veces que el modelo vera todo el dataset recomendado a 50 o mas
imgsz : Tamaño de entrada de las imágenes
batch : Número de imágenes procesadas en paralelo en cada paso recomendado 16 
project="outputs" y name="train_run"
→ Controlan dónde se guardan los resultados.

probandoModelo.py
conf=0.2  <- esta viene a ser la confianza 
