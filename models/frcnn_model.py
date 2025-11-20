"""
CropHealth Detection - Faster R-CNN ResNet50+FPN
Backbone: ResNet50 + Feature Pyramid Network (800x800)
Référence: Fuentes et al. (2017)
"""
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn_model(num_classes: int):
    """
    Construit Faster R-CNN ResNet50-FPN avec poids COCO pré-entraînés
    
    Args:
        num_classes: nombre de classes (10 = 9 classes + background)
    
    Returns:
        model Faster R-CNN prêt pour fine-tuning
    """
    # Charger poids pré-entraînés COCO V2
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # Remplacer le classifieur pour num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model