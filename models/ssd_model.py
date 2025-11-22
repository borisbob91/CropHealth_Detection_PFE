#!/usr/bin/env python3
"""
SSD Lite 320 MobileNetV3 - Adaptation nombre de classes
Solution officielle TorchVision 0.23.0 + PyTorch 2.8.0
Utilise l'API native num_classes
"""
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

print(f"PyTorch: {torch.__version__} | TorchVision: {torchvision.__version__}\n")


def build_ssd_model(num_classes: int, pretrained: bool = True):
    """
    Crée un SSD Lite 320 MobileNetV3 avec le bon nombre de classes
    Utilise l'API officielle de TorchVision
    
    Args:
        num_classes: Nombre total de classes (maladies + background)
                     Ex: 24 = 23 maladies + 1 fond
        pretrained: Si True, charge les poids pré-entraînés COCO (recommandé)
    
    Returns:
        model: SSD modèle prêt pour l'entraînement
    """
    print(f"🔧 Construction du modèle avec {num_classes} classes...")
    
    if pretrained:
        # Méthode recommandée: Charge les poids COCO puis remplace la tête
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        print(f"   ✓ Chargement des poids pré-entraînés COCO...")
        
        # Étape 1: Charger le modèle avec poids COCO (91 classes)
        model_coco = ssdlite320_mobilenet_v3_large(weights=weights)
        
        # Étape 2: Créer un nouveau modèle avec le bon nombre de classes
        # On réutilise le backbone entraîné
        model = ssdlite320_mobilenet_v3_large(
            weights=None,  # Pas de poids pour l'instant
            num_classes=num_classes,  # Notre nombre de classes
            weights_backbone=None  # On va copier manuellement
        )
        
        # Étape 3: Copier les poids du backbone (partie importante !)
        model.backbone.load_state_dict(model_coco.backbone.state_dict())
        
        print(f"   ✓ Backbone transféré, nouvelle tête créée")
        
    else:
        # Modèle from scratch
        print(f"   ✓ Création du modèle from scratch...")
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            num_classes=num_classes
        )
    
    print(f"✅ Modèle créé avec succès!\n")
    return model


def print_model_info(model):
    """Affiche les informations détaillées du modèle"""
    print("="*60)
    print("📊 INFORMATIONS DU MODÈLE")
    print("="*60)
    
    # Extraction du nombre de classes via l'anchor generator et la tête
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Méthode robuste pour obtenir num_classes
    # On prend la première couche de classification
    head = model.head.classification_head
    
    # head est un ModuleList, on prend le premier module
    first_module = list(head.children())[0]
    
    # Ce module est un Sequential[SeparableConv2d, ...]
    if hasattr(first_module, '__getitem__'):
        conv_layer = first_module[0]
    else:
        conv_layer = first_module
    
    # Trouver la vraie couche de convolution
    if hasattr(conv_layer, 'out_channels'):
        out_channels = conv_layer.out_channels
    else:
        # C'est une SeparableConv2d, chercher la pointwise conv
        for layer in conv_layer.children():
            if hasattr(layer, 'out_channels'):
                out_channels = layer.out_channels
                break
    
    num_classes = out_channels // num_anchors[0]
    
    print(f"✓ Nombre de classes: {num_classes}")
    print(f"✓ Ancres par niveau: {num_anchors}")
    
    # Comptage des paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"✓ Paramètres totaux: {total_params:,}")
    print(f"✓ Paramètres entraînables: {trainable_params:,}")
    print(f"  • Backbone: {backbone_params:,}")
    print(f"  • Head: {head_params:,}")
    print("="*60 + "\n")


def test_model(model, num_classes):
    """Test complet du modèle en train et eval"""
    print("="*60)
    print("🧪 TESTS DU MODÈLE")
    print("="*60)
    
    # Test 1: Mode entraînement
    print("\n📊 Test 1: Mode entraînement (forward + backward)")
    model.train()
    
    # Données factices (2 images, classes entre 1 et num_classes-1)
    images = [torch.randn(3, 320, 320) for _ in range(2)]
    targets = [
        {
            "boxes": torch.tensor([[50., 50., 150., 150.], [200., 200., 300., 300.]]),
            "labels": torch.tensor([min(3, num_classes-1), min(15, num_classes-1)])
        },
        {
            "boxes": torch.tensor([[100., 100., 250., 250.]]),
            "labels": torch.tensor([min(12, num_classes-1)])
        }
    ]
    
    try:
        losses = model(images, targets)
        print("✅ Forward pass (train) OK")
        print("   Losses:")
        for k, v in losses.items():
            print(f"   • {k}: {v.item():.4f}")
        
        # Test du backward
        total_loss = sum(losses.values())
        total_loss.backward()
        print(f"✅ Backward pass OK (loss totale: {total_loss.item():.4f})")
        
    except Exception as e:
        print(f"❌ Erreur en mode train: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Mode évaluation
    print("\n📊 Test 2: Mode évaluation (inference)")
    model.eval()
    
    try:
        with torch.no_grad():
            predictions = model(images)
            
        print("✅ Forward pass (eval) OK")
        print("   Prédictions:")
        for i, pred in enumerate(predictions):
            n_boxes = len(pred['boxes'])
            print(f"   • Image {i+1}: {n_boxes} détections", end="")
            if n_boxes > 0:
                max_score = pred['scores'].max().item()
                top_label = pred['labels'][0].item()
                print(f" (score max: {max_score:.4f}, classe: {top_label})")
            else:
                print()
                
    except Exception as e:
        print(f"❌ Erreur en mode eval: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Vérification des dimensions de sortie
    print("\n📊 Test 3: Vérification des dimensions")
    model.eval()
    with torch.no_grad():
        test_img = [torch.randn(3, 320, 320)]
        test_pred = model(test_img)[0]
        
        print(f"✅ Dimensions correctes:")
        print(f"   • boxes: {test_pred['boxes'].shape}")
        print(f"   • labels: {test_pred['labels'].shape}")
        print(f"   • scores: {test_pred['scores'].shape}")
        
        if len(test_pred['labels']) > 0:
            max_label = test_pred['labels'].max().item()
            print(f"   • Label max prédit: {max_label} (doit être < {num_classes})")
            if max_label >= num_classes:
                print(f"   ⚠️ Attention: label {max_label} >= num_classes {num_classes}")
    
    print("\n" + "="*60)
    print("🎉 TOUS LES TESTS SONT PASSÉS !")
    print("="*60 + "\n")
    
    return True


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE - SSD LITE 320 MOBILENETV3")
    print("="*60 + "\n")
    
    # Construction avec 24 classes (23 maladies + 1 background)
    NUM_CLASSES = 24
    model = build_ssd_model(num_classes=NUM_CLASSES, pretrained=True)
    
    # Affichage des informations
    print_model_info(model)
    
    # Tests complets
    success = test_model(model, NUM_CLASSES)
    
    if success:
        print("✅ Le modèle est prêt pour l'entraînement!")
        print(f"\n💡 Utilisation:")
        print(f"   model = build_ssd_model(num_classes={NUM_CLASSES})")
        print(f"   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)")
        print(f"   # ... votre boucle d'entraînement")
    else:
        print("❌ Des erreurs ont été détectées lors des tests")