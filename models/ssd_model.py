"""
CropHealth Detection - SSD MobileNetV3
Backbone: MobileNetV3 Large (320x320)
Référence: Saleem et al. (2019)
"""
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights


def build_ssd_model(num_classes: int):
    """
    Construit SSDLite320 MobileNetV3 avec poids COCO pré-entraînés
    
    Args:
        num_classes: nombre de classes (10 = 9 classes + background)
    
    Returns:
        model SSD prêt pour fine-tuning
    """
    # Charger poids pré-entraînés COCO
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Remplacer le classifieur pour num_classes
    in_channels = model.head.classification_head.module_list[0].in_channels
    num_anchors = model.head.classification_head.module_list[0].out_channels // weights.meta['categories'].__len__()
    
    model.head.classification_head.module_list[0] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.module_list[1] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.module_list[2] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.module_list[3] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.module_list[4] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.module_list[5] = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    
    return model