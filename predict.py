from ultralytics import YOLO
import cv2
import numpy as np

# Charger votre modèle TFLite exporté
model = YOLO('/content/best.tflite')  # ou best_saved_model/

# Vérification
print(f"Classes: {model.names}")
print(f"Taille d'image: {model.args['imgsz']}")

# Sur une image
results = model('/chemin/vers/image.jpg')

# Afficher les résultats
for r in results:
    im_array = r.plot()  # Image avec bounding boxes
    cv2.imwrite('resultat.jpg', im_array)
    print(f"Détections: {len(r.boxes)} objets")
    
    # Détails des détections
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  {model.names[cls]}: {conf:.2f}")