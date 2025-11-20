"""
CropHealth Detection - Entraînement SSD MobileNetV3
Script complet avec early stopping, augmentation et sauvegarde
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import yaml


from datasets.pascalvoc_dataset import PascalVOCDataset
from models.ssd_model import build_ssd_model 
from configs.model_configs import CLASS_NAMES, NUM_CLASSES

# 2️⃣ CONFIGURATION
def get_config():
    """Configuration de l'entraînement SSD"""
    return {
        # Chemins
        'data_root': Path('data/crophealth'),
        'train_dir': 'train',
        'val_dir': 'val',
        'test_dir': 'test',
        
        
        # Hyperparamètres SSD
        'num_epochs': 2,
        'batch_size': 8,  # SSD plus léger → batch size plus grand
        'learning_rate': 0.01,  # SSD utilise typiquement un LR plus élevé
        'weight_decay': 0.0005,
        'momentum': 0.9,
        
        # Early stopping
        'early_stopping_patience': 10,  # epochs sans amélioration
        'early_stopping_min_delta': 0.001,
        
        # Transformations
        'image_size': 320,  # Taille SSD
        
        # Sauvegarde
        'save_dir': Path('outputs/ssd_mobilenetv3'),
        'save_every': 5,
    }


# 3️⃣ TRANSFORMATIONS
def get_transforms(train=True, image_size=320):
    """Pipeline pour SSD (320x320)"""
    if train:
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_area=25,
            min_visibility=0.3
        ))
    else:
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
    
    return transform


# 4️⃣ COLLATE FUNCTION
def collate_fn(batch):
    """Empile les batchs SSD"""
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


class EarlyStopping:
    """Classe de gestion de l'early stopping"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def restore_best(self, model):
        """Restaure les meilleurs poids"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        return model


def evaluate(model, val_loader, device):
    """Évaluation sur validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches



def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """Une epoch d'entraînement"""
    model.train()
    train_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
    
    for images, targets in pbar:
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
        
        # Mise à jour barre de progression
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return train_loss / num_batches


# 8️⃣ SCRIPT PRINCIPAL
def main():
    """Boucle d'entraînement principale"""
    global config
    config = get_config()
    
    # Créer dossier de sauvegarde
    config['save_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Nombre de classes
    num_classes = Nu
    print(f"📦 Nombre de classes: {num_classes} (incl. background)")
    
    # Modèle
    print("📦 Chargement du modèle SSD...")
    model = build_ssd_model(num_classes)
    model.to(device)
    
    # Datasets
    print("📂 Préparation des datasets...")
    train_dataset = PascalVOCDataset(
        img_root=config['data_root'] / config['train_dir'] / 'images',
        ann_root=config['data_root'] / config['train_dir'] / 'Annotations',
        class_names=config['class_names'],
        transforms=get_transforms(train=True, image_size=config['image_size'])
    )
    
    val_dataset = PascalVOCDataset(
        img_root=config['data_root'] / config['val_dir'] / 'images',
        ann_root=config['data_root'] / config['val_dir'] / 'Annotations',
        class_names=config['class_names'],
        transforms=get_transforms(train=False, image_size=config['image_size'])
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📊 Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    
    # Optimiseur (SSD utilise SGD avec momentum élevé)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        restore_best_weights=True
    )
    
    # Historique
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    # Boucle d'entraînement
    print("🚀 Début de l'entraînement SSD...")
    print(f"🛑 Early stopping patience: {config['early_stopping_patience']} epochs")
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        # Scheduler
        scheduler.step()
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\n📈 Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f} | ES Counter: {early_stopping.counter}/{early_stopping.patience}")
        
        # Sauvegarde meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = config['save_dir'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_names': config['class_names']
            }, best_path)
            print(f"💾 Meilleur modèle sauvegardé: {best_path}")
        
        # Sauvegarde périodique
        if epoch % config['save_every'] == 0:
            checkpoint_path = config['save_dir'] / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'class_names': config['class_names']
            }, checkpoint_path)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\n🛑 Early stopping déclenché après {epoch} epochs!")
            print(f"📉 Meilleure loss validation: {early_stopping.best_loss:.4f}")
            model = early_stopping.restore_best(model)
            break
    
    # Sauvegarde finale
    final_path = config['save_dir'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Entraînement terminé! Modèle final sauvegardé: {final_path}")


if __name__ == '__main__':
    main()