"""
CropHealth Detection - Albumentations Transforms
Pipeline unique pour SSD, EfficientDet, Faster R-CNN (classic & light)
YOLOv8n utilise les augmentations natives d'Ultralytics
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_albu_transform(model_name: str, train: bool = True, input_size:int=None):
    """
    Pipeline Albumentations selon modèle

    Args:
        model_name: 'ssd' | 'efficientdet' | 'fasterrcnn' | 'fasterrcnn_light'
        train: True pour augmentations, False pour validation
    
    Returns:
        A.Compose avec bbox_params format pascal_voc
    """
    # Taille d'entrée selon Tableau 6
    size_map = {
        'ssd': 320 if input_size is None else input_size,
        'efficientdet': 512 if input_size is None else input_size,
        'fasterrcnn': 800 if input_size is None else input_size,
        'fasterrcnn_light': 320 if input_size is None else input_size,
    }
    size = size_map[model_name]
    
    # Augmentations communes (si train)
    aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    ] if train else []
    
    # Pipeline finale
    pipeline = A.Compose(
        [
            #*aug,
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )
    return pipeline