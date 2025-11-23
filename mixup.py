"""
MixUp Augmentation pour la Détection d'Objets
Fusionne deux images et leurs annotations de manière aléatoire
"""
import torch
import numpy as np
import random
from typing import List, Dict, Tuple


class MixUpDetection:
    """
    Applique MixUp sur des images de détection d'objets.
    
    MixUp crée une nouvelle image en mélangeant deux images:
        mixed_image = lambda * image1 + (1 - lambda) * image2
        mixed_boxes = boxes1 + boxes2 (concaténation)
    
    Args:
        alpha (float): Paramètre de la distribution Beta. Plus il est élevé,
                      plus le mélange est uniforme. Typique: 1.5 pour détection
        prob (float): Probabilité d'appliquer MixUp [0, 1]
    """
    
    def __init__(self, alpha: float = 1.5, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self, 
        images: List[torch.Tensor], 
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Applique MixUp sur un batch d'images
        
        Args:
            images: Liste de tenseurs [C, H, W]
            targets: Liste de dicts avec 'boxes' [N, 4] et 'labels' [N]
        
        Returns:
            images_mixed: Liste d'images mixées
            targets_mixed: Liste de targets mixés
        """
        if random.random() > self.prob:
            return images, targets
        
        batch_size = len(images)
        images_mixed = []
        targets_mixed = []
        
        # Mélanger aléatoirement les indices
        indices = list(range(batch_size))
        random.shuffle(indices)
        
        for i in range(batch_size):
            # Tirer lambda depuis Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Indices des deux images à mélanger
            idx1 = i
            idx2 = indices[i]
            
            # Mélanger les images
            img1 = images[idx1]
            img2 = images[idx2]
            mixed_img = lam * img1 + (1 - lam) * img2
            
            # Combiner les bounding boxes et labels
            boxes1 = targets[idx1]['boxes']
            labels1 = targets[idx1]['labels']
            boxes2 = targets[idx2]['boxes']
            labels2 = targets[idx2]['labels']
            
            # Concaténer les boxes et labels
            mixed_boxes = torch.cat([boxes1, boxes2], dim=0)
            mixed_labels = torch.cat([labels1, labels2], dim=0)
            
            # Créer le nouveau target
            mixed_target = {
                'boxes': mixed_boxes,
                'labels': mixed_labels
            }
            
            # Ajouter d'autres clés si présentes
            for key in targets[idx1].keys():
                if key not in ['boxes', 'labels']:
                    mixed_target[key] = targets[idx1][key]
            
            images_mixed.append(mixed_img)
            targets_mixed.append(mixed_target)
        
        return images_mixed, targets_mixed


class MixUpCollate:
    """
    Collate function qui applique MixUp directement dans le DataLoader
    Combine avec votre collate_fn existante
    """
    
    def __init__(self, mixup: MixUpDetection, base_collate_fn=None):
        self.mixup = mixup
        self.base_collate_fn = base_collate_fn
    
    def __call__(self, batch):
        # Appliquer le collate de base si fourni
        if self.base_collate_fn:
            images, targets = self.base_collate_fn(batch)
        else:
            images, targets = zip(*batch)
            images = list(images)
            targets = list(targets)
        
        # Appliquer MixUp
        images, targets = self.mixup(images, targets)
        
        return images, targets


def apply_mixup_in_training_loop(
    images: List[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    alpha: float = 1.5,
    prob: float = 0.5
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Version simple pour appliquer MixUp directement dans la boucle d'entraînement
    
    Usage dans train_one_epoch():
        for images, targets in train_loader:
            images, targets = apply_mixup_in_training_loop(images, targets)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            ...
    """
    if random.random() > prob:
        return images, targets
    
    batch_size = len(images)
    lam = np.random.beta(alpha, alpha)
    
    # Mélanger aléatoirement
    indices = torch.randperm(batch_size)
    
    images_mixed = []
    targets_mixed = []
    
    for i in range(batch_size):
        idx1 = i
        idx2 = indices[i].item()
        
        # Mélanger images
        img1 = images[idx1]
        img2 = images[idx2]
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Combiner targets
        mixed_target = {
            'boxes': torch.cat([targets[idx1]['boxes'], targets[idx2]['boxes']], dim=0),
            'labels': torch.cat([targets[idx1]['labels'], targets[idx2]['labels']], dim=0)
        }
        
        images_mixed.append(mixed_img)
        targets_mixed.append(mixed_target)
    
    return images_mixed, targets_mixed


# ============= INTÉGRATION DANS VOTRE CODE =============

def train_one_epoch_with_mixup(model, train_loader, optimizer, device, epoch, config):
    """
    EXEMPLE: Intégration de MixUp dans votre fonction train_one_epoch
    """
    from tqdm.auto import tqdm
    
    model.train()
    train_loss = 0
    num_batches = 0
    
    # Initialiser MixUp
    mixup = MixUpDetection(alpha=1.5, prob=0.5)  # 50% de chance d'appliquer
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
    
    for images, targets in pbar:
        # ✨ APPLIQUER MIXUP AVANT DE PASSER AU GPU
        images, targets = mixup(images, targets)
        
        # Maintenant envoyer au GPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Logging
        train_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return train_loss / num_batches


# ============= MÉTHODE ALTERNATIVE: DANS LE DATALOADER =============

def get_mixup_dataloader(dataset, batch_size, num_workers=4):
    """
    Créer un DataLoader avec MixUp intégré
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    # Créer MixUp collate
    mixup = MixUpDetection(alpha=1.5, prob=0.5)
    mixup_collate = MixUpCollate(mixup, collate_fn)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mixup_collate,
        pin_memory=True
    )


# ============= CONSEILS D'UTILISATION =============
"""
RECOMMANDATIONS:

1. **Alpha parameter**:
   - alpha = 0.2-0.4: Mélange faible (une image domine)
   - alpha = 1.0-1.5: Mélange modéré (RECOMMANDÉ pour détection)
   - alpha = 2.0+: Mélange fort (50/50)

2. **Probabilité**:
   - prob = 0.3-0.5: Bon compromis (RECOMMANDÉ)
   - prob = 1.0: Trop agressif, peut nuire à l'apprentissage

3. **Quand utiliser MixUp**:
   ✅ Dataset petit/moyen (< 10k images)
   ✅ Problème d'overfitting
   ✅ Besoin de régularisation forte
   ❌ Dataset très grand (> 100k images)
   ❌ Objets très petits/denses

4. **Combinaison avec autres augmentations**:
   - Appliquer les augmentations spatiales (rotation, crop) AVANT MixUp
   - MixUp doit être la DERNIÈRE transformation

5. **Performance**:
   - MixUp ralentit légèrement l'entraînement (~5-10%)
   - Amélioration mAP: +1-3% typiquement
   - Meilleure généralisation
"""