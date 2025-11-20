"""
CropHealth Detection - Entraînement EfficientDet-D0
Script 100% fonctionnel avec wrapper
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision

# Imports
from datasets.pascalvoc_dataset import PascalVOCDataset
from models.effdet_model import build_efficientdet_model, EfficientDetWrapper


# 1️⃣ CONFIGURATION
def get_config():
    return {
        # Chemins
        'data_root': Path('data/crophealth'),
        'train_dir': 'train',
        'val_dir': 'val',
        
        # Classes (9 maladies + background)
        'class_names': [
            'aphid', 'armyworm', 'beetle', 'bollworm', 
            'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer'
        ],
        
        # Hyperparamètres
        'num_epochs': 50,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        
        # Early stopping
        'early_stopping_patience': 8,
        'early_stopping_metric': 'map50',
        'early_stopping_mode': 'max',
        
        # Transformations
        'image_size': 512,
        
        # Sauvegarde
        'save_dir': Path('outputs/efficientdet_d0'),
        'log_dir': Path('runs/efficientdet_d0'),
        'save_every': 5,
    }


# 2️⃣ TRANSFORMATIONS (512x512)
def get_transforms(train=True, image_size=512):
    if train:
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=25, min_visibility=0.3))
    else:
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    return transform


# 3️⃣ COLLATE FUNCTION
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# 4️⃣ EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=8, metric='map50', mode='max', min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, current_value, model, epoch):
        improved = (self.mode == 'max' and current_value > self.best_value + self.min_delta) or \
                   (self.mode == 'min' and current_value < self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def restore_best(self, model, device):
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
        return model


# 5️⃣ ÉVALUATION mAP@50
@torch.inference_mode()
def evaluate(model, val_loader, device, writer, epoch):
    """Validation avec mAP@50"""
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy')
    
    val_loss = 0
    num_batches = 0
    
    for imgs, targets in tqdm(val_loader, desc="Validation", leave=False):
        imgs = [img.to(device) for img in imgs]
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Loss
        loss_dict = model(imgs, targets_device)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
        num_batches += 1
        
        # Inference
        preds = model(imgs)
        
        # Formater pour torchmetrics
        preds_formatted = []
        targets_formatted = []
        
        for pred, target in zip(preds, targets):
            if len(pred['boxes']) > 0:  # Filtrer les prédictions vides
                preds_formatted.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu(),
                })
            else:
                preds_formatted.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                })
            
            targets_formatted.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
            })
        
        metric.update(preds_formatted, targets_formatted)
    
    results = metric.compute()
    map50 = results['map_50'].item()
    map_all = results['map'].item()
    avg_val_loss = val_loss / num_batches
    
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    writer.add_scalar('val/mAP50', map50, epoch)
    writer.add_scalar('val/mAP', map_all, epoch)
    
    metric.reset()
    return avg_val_loss, map50, map_all


# 6️⃣ ENTRAÎNEMENT
def train_one_epoch(model, train_loader, optimizer, device, writer, epoch):
    """Entraîne une epoch"""
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
        
        train_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
    
    avg_train_loss = train_loss / num_batches
    writer.add_scalar('train/loss', avg_train_loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return avg_train_loss


# 7️⃣ SCRIPT PRINCIPAL
def main():
    """Boucle d'entraînement"""
    global config
    config = get_config()
    
    # Dossiers
    config['save_dir'].mkdir(parents=True, exist_ok=True)
    config['log_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # TensorBoard
    writer = SummaryWriter(config['log_dir'])
    print(f"📊 TensorBoard: tensorboard --logdir={config['log_dir']}")
    
    # Modèle
    num_classes = len(config['class_names']) + 1
    print(f"📦 Classes: {num_classes}")
    
    model = build_efficientdet_model(num_classes=num_classes)
    model.to(device)
    
    # Datasets
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
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        metric=config['early_stopping_metric'],
        mode=config['early_stopping_mode']
    )
    
    # Boucle
    print(f"🚀 Début entraînement EfficientDet-D0...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, writer, epoch)
        val_loss, map50, map_all = evaluate(model, val_loader, device, writer, epoch)
        
        scheduler.step()
        
        print(f"\n📈 Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")
        print(f"🎯 mAP@50: {map50:.3f} | mAP: {map_all:.3f}")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f} | ES: {early_stopping.counter}/{early_stopping.patience}")
        
        # Sauvegarde meilleur modèle
        current_metric = map50 if config['early_stopping_metric'] == 'map50' else val_loss
        is_best = (config['early_stopping_mode'] == 'max' and current_metric > early_stopping.best_value) or \
                  (config['early_stopping_mode'] == 'min' and current_metric < early_stopping.best_value)
        
        if is_best:
            best_path = config['save_dir'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'map50': map50,
                'map': map_all,
                'class_names': config['class_names']
            }, best_path)
            print(f"💾 Meilleur modèle: {best_path}")
        
        # Checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = config['save_dir'] / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'map50': map50,
                'map': map_all,
                'class_names': config['class_names']
            }, checkpoint_path)
        
        # Early stopping
        early_stopping(current_metric, model, epoch)
        if early_stopping.should_stop:
            print(f"\n🛑 Early stopping à l'epoch {epoch}!")
            model = early_stopping.restore_best(model, device)
            break
    
    # Final
    final_path = config['save_dir'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    
    writer.close()
    print(f"\n✅ Entraînement terminé! Final: {final_path}")


if __name__ == '__main__':
    main()