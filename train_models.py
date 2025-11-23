"""
CropHealth Detection - Entraînement Unifié Multi-Modèles
Supporte: SSD MobileNetV3, Faster R-CNN, Faster R-CNN Light
"""
from datetime import datetime
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Imports locaux
from datasets.pascalvoc_dataset import PascalVOCDataset
from datasets.yolo_dataset import YoloDataset
from early_stopping import EarlyStopping
from metric_logger import AdvancedYoloLogger
from models.ssd_model import build_ssd_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model
from configs.model_configs import CLASS_NAMES, NUM_CLASSES
from train import build_dataloaders
from utils.yolo_style_logger import save_yolo_style_checkpoint


# ═══════════════════════════════════════════════════════════════════
# 📋 CONFIGURATIONS PAR MODÈLE
# ═══════════════════════════════════════════════════════════════════

def get_model_config(model_type='ssd'):
    """
    Retourne la configuration spécifique pour chaque modèle
    
    Args:
        model_type: 'ssd', 'frcnn', ou 'frcnn_light'
    """
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    # Configuration de base commune
    base_config = {
        'data_root': Path(r'C:\Users\BorisBob\Desktop\detection\pascal_test\yolo_test'),
        'train_dir': 'train',
        'val_dir': 'val',
        'test_dir': 'test',
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 0.001,
        'save_every': 5,
    }
    
    # Configurations spécifiques par modèle
    model_configs = {
        'ssd': {
            'name': 'CropHealth_SSD',
            'backbone': 'MobileNetV3',
            'num_epochs': 100,
            'batch_size': 8,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'image_size': 320,
            'input_size': 320,
            'dataset_format': 'yolo',
            'save_dir': Path(f"runs/ssd_mobilenetv3_{timestamp}"),
            'optimizer_type': 'sgd',
            'scheduler_type': 'cosine',
            'scheduler_params': {
                'T_max': 100,
                'eta_min': 1e-6
            }
        },
        
        'frcnn': {
            'name': 'CropHealth_FasterRCNN',
            'backbone': 'ResNet50',
            'num_epochs': 20,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'image_size': 800,
            'input_size': 800,
            'dataset_format': 'pascal_voc',
            'save_dir': Path(f"runs/faster_rcnn_{timestamp}"),
            'optimizer_type': 'sgd',
            'scheduler_type': 'step',
            'scheduler_params': {
                'step_size': 15,
                'gamma': 0.1
            }
        },
        
        'frcnn_light': {
            'name': 'CropHealth_FasterRCNN_Light',
            'backbone': 'MobileNetV3',
            'num_epochs': 30,
            'batch_size': 6,
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'image_size': 320,
            'input_size': 320,
            'dataset_format': 'pascal_voc',
            'save_dir': Path(f"runs/faster_rcnn_light_{timestamp}"),
            'optimizer_type': 'sgd',
            'scheduler_type': 'step',
            'scheduler_params': {
                'step_size': 15,
                'gamma': 0.1
            }
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Model type '{model_type}' non supporté. Choisir parmi: {list(model_configs.keys())}")
    
    # Fusionner config de base avec config spécifique
    config = {**base_config, **model_configs[model_type]}
    config['model_type'] = model_type
    
    return config


# ═══════════════════════════════════════════════════════════════════
# 🏗️ CONSTRUCTION DU MODÈLE
# ═══════════════════════════════════════════════════════════════════

def build_model(model_type, num_classes):
    """Construit le modèle selon le type spécifié"""
    model_builders = {
        'ssd': build_ssd_model,
        'frcnn': build_fasterrcnn_model,
        'frcnn_light': build_fasterrcnn_light_model
    }
    
    if model_type not in model_builders:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    print(f"🔨 Construction du modèle {model_type.upper()}...")
    return model_builders[model_type](num_classes)


# ═══════════════════════════════════════════════════════════════════
# ⚙️ CONSTRUCTION OPTIMISEUR & SCHEDULER
# ═══════════════════════════════════════════════════════════════════

def build_optimizer(model, config):
    """Construit l'optimiseur selon la configuration"""
    optimizer_type = config.get('optimizer_type', 'sgd').lower()
    
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Optimiseur non supporté: {optimizer_type}")
    
    print(f"⚙️ Optimiseur: {optimizer_type.upper()}")
    return optimizer


def build_scheduler(optimizer, config):
    """Construit le scheduler selon la configuration"""
    scheduler_type = config.get('scheduler_type', 'cosine').lower()
    params = config.get('scheduler_params', {})
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', config['num_epochs']),
            eta_min=params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 15),
            gamma=params.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.get('milestones', [10, 20]),
            gamma=params.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Scheduler non supporté: {scheduler_type}")
    
    print(f"📊 Scheduler: {scheduler_type.upper()}")
    return scheduler


# ═══════════════════════════════════════════════════════════════════
# 🔄 TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════

def get_transforms(train=True, image_size=320):
    """Pipeline d'augmentation Albumentations"""
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


# ═══════════════════════════════════════════════════════════════════
# 📈 ÉVALUATION
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def evaluate(model, val_loader, device):
    """
    Validation avec mAP@50
    2 passes: une pour loss (train mode), une pour mAP (eval mode)
    """
    model.eval()
    
    # PASS 1: Calculer la loss en mode train (sans gradients)
    train_loss_epoch = 0
    num_batches = 0
    was_training = model.training
    
    model.train()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss_epoch += losses.item()
            num_batches += 1
    
    val_loss = train_loss_epoch / num_batches if num_batches > 0 else 0.0
    
    # PASS 2: Calculer mAP en mode eval
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy')
    
    for imgs, targets in val_loader:
        imgs = [img.to(device) for img in imgs]
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Inference
        preds = model(imgs)
        
        # Formater pour torchmetrics
        preds_formatted = []
        targets_formatted = []
        
        for pred, target in zip(preds, targets):
            preds_formatted.append({
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu(),
            })
            targets_formatted.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
            })
        
        metric.update(preds_formatted, targets_formatted)
    
    results = metric.compute()
    map50 = results['map_50'].item()
    map_all = results['map'].item()
    
    # Restaurer l'état original
    if was_training:
        model.train()
    
    metric.reset()
    return val_loss, map50, map_all


# ═══════════════════════════════════════════════════════════════════
# 🏋️ ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, optimizer, device, epoch, config):
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
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return train_loss / num_batches


# ═══════════════════════════════════════════════════════════════════
# 🚀 SCRIPT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main(model_type='ssd'):
    """
    Boucle d'entraînement principale
    
    Args:
        model_type: 'ssd', 'frcnn', ou 'frcnn_light'
    """
    # Configuration
    config = get_model_config(model_type)
    print(f"\n{'='*70}")
    print(f"🎯 ENTRAÎNEMENT: {config['name']}")
    print(f"{'='*70}")
    print(f"📋 Configuration:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Créer dossier de sauvegarde
    config['save_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Logger
    vis_class_names = ['background'] + CLASS_NAMES
    logger = AdvancedYoloLogger(
        save_dir=config['save_dir'],
        class_names=vis_class_names,
        device=device
    )
    
    # Nombre de classes
    num_classes = NUM_CLASSES
    print(f"📦 Nombre de classes: {num_classes} (incl. background)")
    
    # Modèle
    model = build_model(model_type, num_classes)
    model.to(device)
    
    # Datasets & DataLoaders
    print("📂 Préparation des datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(
        model_type if model_type == 'ssd' else 'frcnn',
        config['data_root'],
        config
    )
    
    # Optimiseur & Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        restore_best_weights=True
    )
    
    # Historique
    history = {
        'train_loss': [],
        'val_loss': [],
        'map50': [],
        'map': []
    }
    best_metric = 0.0
    
    # Boucle d'entraînement
    print(f"\n🚀 Début de l'entraînement...")
    print(f"🛑 Early stopping patience: {config['early_stopping_patience']} epochs\n")
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Validation
        val_loss, map50, map_all = evaluate(model, val_loader, device)
        
        # Scheduler
        scheduler.step()
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['map50'].append(map50)
        history['map'].append(map_all)
        
        # Affichage
        print(f"\n📈 Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"🎯 mAP@50: {map50:.3f} | mAP: {map_all:.3f}")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f} | ES Counter: {early_stopping.counter}/{early_stopping.patience}")
        
        # Logger
        logger.log_epoch(epoch, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'map50': map50,
            'map': map_all,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Sauvegarde meilleur modèle
        if map50 > best_metric:
            best_metric = map50
            best_path = config['save_dir'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'map50': map50,
                'map': map_all,
                'class_names': CLASS_NAMES,
                'config': config
            }, best_path)
            early_stopping.reset_counter()
            print(f"💾 Meilleur modèle sauvegardé avec mAP@50: {map50:.3f}")
            logger.generate_full_report(model, val_loader, epoch="best")
        
        # Sauvegarde périodique
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
                'class_names': CLASS_NAMES,
                'config': config
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'config': config
    }, final_path)
    
    print(f"\n✅ Entraînement terminé!")
    print(f"💾 Modèle final: {final_path}")
    print(f"🎯 Meilleur mAP@50: {best_metric:.3f}")
    
    # Rapport final
    save_yolo_style_checkpoint(
        model=model,
        val_loader=val_loader,
        epoch=epoch,
        save_dir=config['save_dir'],
        device=device,
        prefix="best"
    )
    logger.generate_full_report(model, val_loader, epoch="FINAL")


# ═══════════════════════════════════════════════════════════════════
# 🎬 POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement unifié multi-modèles')
    parser.add_argument(
        '--model',
        type=str,
        default='ssd',
        choices=['ssd', 'frcnn', 'frcnn_light'],
        help='Type de modèle à entraîner (default: ssd)'
    )
    
    args = parser.parse_args()
    
    print(f"""
    ╔════════════════════════════════════════════════════════╗
    ║       CropHealth Detection - Entraînement Unifié       ║
    ║                  Modèle: {args.model.upper():^20}      ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    main(model_type=args.model)