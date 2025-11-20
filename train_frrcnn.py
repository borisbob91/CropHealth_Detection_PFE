"""
CropHealth Detection - Entraînement Faster R-CNN
Script complet avec augmentation, logging et sauvegarde
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml
from pathlib import Path
import os

# 1️⃣ IMPORTER LES DEUX ÉLÉMENTS PRÉCÉDENTS
# Copiez ici les classes des messages précédents
from datasets.pascalvoc_dataset import PascalVOCDataset
from models.frcnn_model import build_fasterrcnn_model
from configs.model_configs import CLASS_NAMES, NUM_CLASSES

# 2️⃣ CONFIGURATION
def get_config():
    """Configuration de l'entraînement"""
    return {
        # Chemins
        'data_root': Path('C:\\Users\\BorisBob\\Desktop\\detection\\dataset_split\\label_studio\\pascal_voc_orignal'),
        'train_dir': 'train',
        'val_dir': 'val',
        

        # Hyperparamètres
        'num_epochs': 2,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        
        # Transformations
        'image_size': 800,
        
        # Sauvegarde
        'save_dir': Path('./outputs/faster_rcnn'),
        'save_every': 5,  # sauvegarde tous les 5 epochs
    }


# 3️⃣ TRANSFORMATIONS ALBUMENTATIONS
def get_transforms(train=True, image_size=800):
    """Pipeline d'augmentation"""
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
    """Empile les batchs avec des images de tailles différentes"""
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


# 5️⃣ ÉVALUATION
def evaluate(model, val_loader, device):
    """Évaluation rapide sur validation set"""
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


# 6️⃣ FONCTION D'ENTRAÎNEMENT
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """Entraîne pour une epoch"""
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


# 7️⃣ SCRIPT PRINCIPAL
def main():
    from configs.model_configs import CLASS_NAMES, NUM_CLASSES
    """Boucle d'entraînement complète"""
    global config
    config = get_config()
    
    # Créer dossier de sauvegarde
    config['save_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Nombre de classes (9 + background)
    num_classes = NUM_CLASSES
    
    # Modèle
    print("📦 Chargement du modèle...")
    model = build_fasterrcnn_model(num_classes)
    model.to(device)
    
    # Datasets
    print("📂 Préparation des datasets...")
    train_dataset = PascalVOCDataset(
        img_root=config['data_root'] / config['train_dir'] / 'images',
        ann_root=config['data_root'] / config['train_dir'] / 'Annotations',
        class_names=CLASS_NAMES,
        transforms=get_transforms(train=True, image_size=config['image_size'])
    )
    
    val_dataset = PascalVOCDataset(
        img_root=config['data_root'] / config['val_dir'] / 'images',
        ann_root=config['data_root'] / config['val_dir'] / 'Annotations',
        class_names=CLASS_NAMES,
        transforms=get_transforms(train=False, image_size=config['image_size'])
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
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
    
    # Optimiseur
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )
    
    # Historique
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    # Boucle d'entraînement
    print("🚀 Début de l'entraînement...")
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
        
        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = config['save_dir'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_names': CLASS_NAMES
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
                'class_names': CLASS_NAMES
            }, checkpoint_path)
    
    # Sauvegarde finale
    final_path = config['save_dir'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Entraînement terminé! Modèle final sauvegardé: {final_path}")


if __name__ == '__main__':
    main()