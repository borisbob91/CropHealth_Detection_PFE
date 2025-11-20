"""
CropHealth Detection - Faster R-CNN MobileNetV3+FPN (light)
Backbone: MobileNetV3 Large + Feature Pyramid Network (320x320)
Référence: Fuentes et al. (2017) - version légère
"""
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn_light_model(num_classes: int):
    """
    Construit Faster R-CNN MobileNetV3-FPN avec poids COCO pré-entraînés
    
    Args:
        num_classes: nombre de classes (10 = 9 classes + background)
    
    Returns:
        model Faster R-CNN light prêt pour fine-tuning
    """
    # Charger poids pré-entraînés COCO
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    
    # Remplacer le classifieur pour num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model