from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO con el archivo que ya tienes
model = YOLO("yolo11x.pt")

# Realizar la predicción sobre el video
results = model.predict(source="2.mp4", show=True, stream=True, verbose=False)

# Iterar sobre cada frame y obtener las detecciones
for result in results:
    # Obtener las clases detectadas en el frame
    detections = result.boxes.cls  # Detecciones de clases

    # Contar cuántas personas (clase 0) hay en este frame
    person_count = (detections == 0).sum()  # Clase '0' generalmente es 'person'
    
    if(person_count>2):
      print("Se detectaron mas de 2 personas")
    else:
      print(f"Solo {person_count} personas detectadas")
