"""
CropHealth Detection - Albumentations Transforms
Pipeline unique pour SSD, EfficientDet, Faster R-CNN (classic & light)
YOLOv8n utilise les augmentations natives d'Ultralytics
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import warnings
from typing import List, Dict, Optional, Tuple


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



def validate_bbox_format(boxes: List[List[float]], img_shape: Tuple[int, int], 
                         filename: str = "unknown") -> Optional[torch.Tensor]:
    """
    Valide si les boîtes sont normalisées (0-1) ou en pixels.
    Si non normalisées, affiche un message et retourne None.
    
    Args:
        boxes: Liste de boîtes [x1, y1, x2, y2]
        img_shape: (height, width) pour vérification si nécessaire
        filename: Nom du fichier pour le message d'erreur
    
    Returns:
        Tensor des boîtes si valides, None sinon
    """
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    
    # Convertir en tensor pour vérification
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    
    # Vérifier si les valeurs sont déjà normalisées (entre 0 et 1)
    if boxes_tensor.max() <= 1.0:
        # ✅ Normalisé - accepter
        return boxes_tensor
    
    # Vérifier si c'est des coordonnées en pixels mais valides
    h, w = img_shape
    if boxes_tensor.max() <= max(h, w):
        # ⚠️ En pixels mais valides - avertir et ignorer
        warnings.warn(
            f"⚠️ Boîtes en PIXELS détectées dans {filename}: "
            f"max={boxes_tensor.max()} > 1.0. "
            f"Ignorées. Normalisez vos annotations (x/W, y/H)."
        )
        return None
    
    # ❌ Valeurs invalides (trop grandes ou négatives)
    warnings.warn(
        f"❌ Boîtes INVALIDES détectées dans {filename}: "
        f"max={boxes_tensor.max()}, min={boxes_tensor.min()}. "
        f"Ignorées."
    )
    return None