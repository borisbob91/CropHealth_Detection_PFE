"""
CropHealth Detection - EfficientDet-D0
Backbone: EfficientNet-B0 + BiFPN (512x512)
Version CORRIGÉE avec wrapper compatible torchvision
"""
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
import sys

def build_efficientdet_model(num_classes: int, pretrained: bool = True):
    """
    Construit EfficientDet-D0 avec API compatible torchvision
    
    Args:
        num_classes: Nombre TOTAL de classes (ex: 10 = 9 classes + background)
        pretrained: Charger les poids COCO
    
    Returns:
        Model avec API .train() / .eval() compatible
    """
    try:
        from effdet import create_model as create_effdet_model
        
        # Créer le modèle en mode entraînement
        model = create_effdet_model(
            'tf_efficientdet_d0',
            bench_task='train',  # Mode entraînement pour la loss
            num_classes=num_classes,
            pretrained=pretrained,
            checkpoint_path=None,
            image_size=(512, 512),
        )
        
        # WRAPPER CRITIQUE : Rend l'API compatible torchvision
        model = EfficientDetWrapper(model, num_classes)
        
        print(f"✅ EfficientDet-D0 créé | {num_classes} classes | {count_params(model):.1f}M paramètres")
        return model
        
    except ImportError:
        raise RuntimeError(
            "❌ effdet non installé. Exécutez: pip install effdet timm\n"
            " Ou utilisez l'alternative YOLOv5/v8 plus simple."
        )


class EfficientDetWrapper(nn.Module):
    """
    Wrapper qui rend EfficientDet compatible avec l'API torchvision
    - Mode train: retourne loss_dict
    - Mode eval: retourne predictions
    """
    def __init__(self, effdet_model, num_classes):
        super().__init__()
        self.model = effdet_model
        self.num_classes = num_classes
        self.training = True  # Mode par défaut
    
    def train(self, mode: bool = True):
        """Basculer entre mode entraînement et inférence"""
        self.training = mode
        self.model.bench.eval()  # Toujours en eval pour le bench
        return self
    
    def eval(self):
        """Mode évaluation"""
        return self.train(False)
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict]] = None) -> Union[Dict, List[Dict]]:
        """
        API compatible:
        - Train: images + targets → loss_dict
        - Eval: images → predictions
        """
        if self.training:
            # MODE ENTRAÎNEMENT : Calculer la loss
            if targets is None:
                raise ValueError("En mode entraînement, 'targets' est requis")
            
            # effdet attend des targets formatés différemment
            # On utilise directement le bench d'entraînement
            loss_dict = self.model(images, targets)  # DetBenchTrain.forward
            return loss_dict
        
        else:
            # MODE ÉVALUATION : Prédiction seule
            # DetBenchPredict fait tout le travail (NMS, scores, etc.)
            predictions = self.model(images)  # DetBenchPredict.forward
            return predictions
    
    @property
    def config(self):
        """Accès à la config"""
        return self.model.config
    
    def state_dict(self):
        """Exposer les poids"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Charger les poids"""
        return self.model.load_state_dict(state_dict)


def count_params(model):
    """Compte les paramètres en millions"""
    return sum(p.numel() for p in model.parameters()) / 1e6


# ==================== TEST RAPIDE ====================
if __name__ == "__main__":
    # Test 1: Mode entraînement
    model = build_efficientdet_model(num_classes=10)
    model.train()
    
    x = [torch.randn(3, 512, 512) for _ in range(2)]  # Batch de 2 images
    targets = [
        {'boxes': torch.tensor([[10, 10, 50, 50]]), 'labels': torch.tensor([1])},
        {'boxes': torch.tensor([[30, 30, 100, 100]]), 'labels': torch.tensor([2])}
    ]
    
    # Test forward train
    loss_dict = model(x, targets)
    print(f"✅ Mode train OK | Loss: {loss_dict['loss'].item():.4f}")
    
    # Test 2: Mode évaluation
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        print(f"✅ Mode eval OK | {len(predictions)} prédictions")
        if predictions[0]:
            print(f"   Boxes: {predictions[0]['boxes'].shape} | Scores: {predictions[0]['scores'].shape}")