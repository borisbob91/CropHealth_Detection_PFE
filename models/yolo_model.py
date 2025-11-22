"""
CropHealth Detection - YOLOv11n
Backbone: CSP-Darknet (640x640)
Via Ultralytics framework
"""
from ultralytics import YOLO


def build_yolo_model(num_classes: int, weights='yolo11n.pt'):
    """
    Construit YOLOv11n avec poids COCO pré-entraînés
    
    Args:
        num_classes: nombre de classes (9 dans notre cas, pas besoin de +1 pour YOLO)
        weights: chemin vers poids pré-entraînés ou 'yolo11n.pt'
    
    Returns:
        model YOLO prêt pour fine-tuning
    """
    # Charger modèle pré-entraîné
    model = YOLO(weights)
    
    # Le nombre de classes sera configuré automatiquement via data.yaml
    # lors du training
    
    return model


# Test rapide
if __name__ == "__main__":
    model = build_yolo_model(num_classes=9)
    print(f"✅ YOLOv11n chargé avec succès")
    print(f"   • Model: {model.model}")