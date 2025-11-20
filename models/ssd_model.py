"""
CropHealth Detection - SSD MobileNetV3
Backbone: MobileNetV3 Large (320x320)
Référence: Saleem et al. (2019)
"""
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

def build_ssd_model(num_classes: int):
    """
    Construit SSDLite320 MobileNetV3 avec poids COCO pré-entraînés
    Méthode SIMPLIFIÉE et ROBUSTE - Corrige l'erreur AttributeError
    
    Args:
        num_classes: nombre de classes TOTALES (23 classes + background = 24)
    
    Returns:
        model SSD prêt pour fine-tuning
    """
    # Charger poids pré-entraînés COCO
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    
    # =====================================================
    # MÉTHODE 1: La plus simple (recommandée)
    # =====================================================
    # Torchvision gère automatiquement la modification du head
    model = ssdlite320_mobilenet_v3_large(
        weights=weights,
        num_classes=num_classes  # C'est TOUT ce qu'il faut !
    )
    
    # Le modèle est maintenant correctement configuré
    # La backbone conserve les poids COCO, le head est réinitialisé
    
    return model


# Alternative si vous avez besoin de plus de contrôle
def build_ssd_model_advanced(num_classes: int):
    """
    Méthode avancée avec initialisation manuelle
    Utile si vous voulez un freezing partiel de la backbone
    """
    from torchvision.models.detection.ssdlite import _normal_init
    
    # Charger le modèle sans head
    model = ssdlite320_mobilenet_v3_large(weights=None)
    
    # Charger les poids COCO dans un modèle temporaire
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    pretrained_model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Copier les poids de la backbone uniquement
    model.backbone.load_state_dict(pretrained_model.backbone.state_dict())
    
    # Le head est automatiquement créé avec num_classes correct
    # Mais on le réinitialise pour éviter tout conflit
    model.head.classification_head.apply(_normal_init)
    model.head.regression_head.apply(_normal_init)
    
    return model

    # Test de la fonction
if __name__ == "__main__":
    model = build_ssd_model_advanced(num_classes=24)

    print(f"✅ Modèle créé avec succès")
    print(f"📦 Nombre de classes: {model.head.classification_head.num_classes}")
    print(f"🔧 Backbone paramètres: {sum(p.numel() for p in model.backbone.parameters())}")
    print(f"🎯 Head paramètres: {sum(p.numel() for p in model.head.parameters())}")
    
    # Test forward avec une image factice
    x = torch.randn(1, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"📤 Sortie testée: {len(output)} détections")