"""
CropHealth Detection - EfficientDet-D0
Backbone: EfficientNet-B0 + BiFPN (512x512)
Référence: Tan et al. (2020)
"""
from effdet import create_model
from effdet.config import get_efficientdet_config


def build_efficientdet_model(num_classes: int):
    """
    Construit EfficientDet-D0 avec poids COCO pré-entraînés
    
    Args:
        num_classes: nombre de classes (10 = 9 classes + background)
    
    Returns:
        model EfficientDet-D0 prêt pour fine-tuning
    """
    # Configuration EfficientDet-D0
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = 512
    
    # Charger poids pré-entraînés COCO via timm
    model = create_model(
        'tf_efficientdet_d0',
        bench_task='train',
        num_classes=num_classes,
        pretrained=True,
        checkpoint_path='',
        image_size=512,
    )
    
    return model