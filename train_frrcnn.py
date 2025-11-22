"""
CropHealth Detection - Entraînement Faster R-CNN
VERSION CORRIGÉE - Gestion propre de la validation
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision

# Imports
from datasets.pascalvoc_dataset import PascalVOCDataset
from models.frcnn_model import build_fasterrcnn_model
from configs.model_configs import CLASS_NAMES, NUM_CLASSES


# 1️⃣ CONFIGURATION
def get_config():
    return {
        'data_root': Path('/content/CropHealth_Detection_PFE/data/pascal_voc_original'),
        'train_dir': 'train',
        'val_dir': 'val',
        'num_epochs': 20,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'image_size': 800,
        'save_dir': Path('./outputs/faster_rcnn'),
        'save_every': 5,
    }


# 2️⃣ TRANSFORMATIONS
def get_transforms(train=True, image_size=800):
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


# 4️⃣ ÉVALUATION (CORRIGÉE PROPREMENT)
@torch.inference_mode()
def evaluate(model, val_loader, device):
    """
    Validation avec mAP@50 - VERSION PROPRE
    On utilise 2 passes : une pour loss (train), une pour mAP (eval)
    """
    model.eval()  # Mode évaluation pour inference

    # 🔥 PASS 1: Calculer la loss en mode train mais SANS gradients
    train_loss_epoch = 0
    num_batches = 0

    # On sauvegarde l'état original
    was_training = model.training

    model.train()  # Passer temporairement en mode train
    with torch.no_grad():  # Mais SANS calculer les gradients
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss_epoch += losses.item()
            num_batches += 1

    val_loss = train_loss_epoch / num_batches if num_batches > 0 else 0.0

    # 🔥 PASS 2: Calculer mAP en mode eval
    model.eval()  # Revenir en mode eval pour inference
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


# 5️⃣ ENTRAÎNEMENT
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        num_batches += 1

        pbar.set_postfix({'loss': f"{losses.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

    return train_loss / num_batches


# 6️⃣ SCRIPT PRINCIPAL
def main():
    global config
    config = get_config()

    # Dossiers
    config['save_dir'].mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Modèle
    num_classes = NUM_CLASSES
    print(f"📦 Classes: {num_classes}")

    model = build_fasterrcnn_model(num_classes)
    model.to(device)

    # Datasets
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

    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Optimiseur
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Historique
    history = {'train_loss': [], 'val_loss': [], 'map50': [], 'map': []}
    best_metric = 0.0

    # Boucle
    print("🚀 Début de l'entraînement...")

    for epoch in range(1, config['num_epochs'] + 1):
        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

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
        print(f"\n📈 Epoch {epoch}: Train={train_loss:.4f} | Val Loss={val_loss:.4f}")
        print(f"🎯 mAP@50: {map50:.3f} | mAP: {map_all:.3f}")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f}")

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
                'class_names': CLASS_NAMES
            }, best_path)
            print(f"💾 Meilleur modèle sauvegardé: {best_path}")

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
                'class_names': CLASS_NAMES
            }, checkpoint_path)

    # Final
    final_path = config['save_dir'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Entraînement terminé! Final: {final_path}")


if __name__ == '__main__':
    main()