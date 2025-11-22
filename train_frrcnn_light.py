import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision

# 1️⃣ IMPORTS
from datasets.pascalvoc_dataset import PascalVOCDataset
from models.frcnn_light_model import build_fasterrcnn_light_model
from configs.model_configs import CLASS_NAMES, NUM_CLASSES

# 2️⃣ CONFIGURATION
def get_config():
    """Configuration optimisée"""
    return {
        # Chemins
        'data_root': Path('/content/CropHealth_Detection_PFE/data/pascal_voc_original'),
        'train_dir': 'train',
        'val_dir': 'val',
        'test_dir': 'test',

        # Hyperparamètres
        'num_epochs': 30,
        'batch_size': 6,
        'learning_rate': 0.005,
        'weight_decay': 0.0005,
        'momentum': 0.9,

        # Early stopping (sur mAP50 cette fois)
        'early_stopping_patience': 8,
        'early_stopping_metric': 'map50',  # 'loss' ou 'map50'
        'early_stopping_mode': 'max',      # 'min' pour loss, 'max' pour mAP

        # Transformations
        'image_size': 320,

        # Sauvegarde et logging
        'save_dir': Path('outputs/faster_rcnn_light'),
        'save_every': 5,
        'log_dir': Path('runs/faster_rcnn_light'),
    }


# 3️⃣ TRANSFORMATIONS
def get_transforms(train=True, image_size=320):
    """Pipeline Albumentations"""
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
            min_area=10,
            min_visibility=0.2
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
    """Empile les batchs et filtre les images sans bounding boxes"""
    # Filtrer les échantillons où target['boxes'] est vide
    batch = [item for item in batch if len(item[1]['boxes']) > 0]
    if not batch:
        return None, None # Retourner None si le batch est vide après filtrage

    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets

def collate_fn_test(batch):
    """
    Filtre les images sans boîtes de détection avant de former le lot (batch).
    """
    # Votre batch est une liste de tuples [(image1, target1), (image2, target2), ...]

    # 1. Filtrer les échantillons où target['boxes'] est vide ou mal formé
    valid_batch = []
    for image, target in batch:
        # Vérifie que les boîtes existent ET qu'elles ont la bonne dimension
        # La condition est typique si 'target' contient des clés 'boxes' et 'labels'
        if 'boxes' in target and target['boxes'].numel() > 0:
            valid_batch.append((image, target))
        # Note: Si le target['boxes'] est un torch.Size([0]), il faut s'assurer
        # qu'il soit bien formaté en torch.Size([0, 4]) dans la classe Dataset elle-même.

    if not valid_batch:
        # Si tout le lot est invalide après le filtrage, on renvoie une liste vide
        # ou on lève une exception (moins recommandé).
        # Pour des raisons de stabilité, c'est souvent mieux de laisser le loader sauter ce lot.
        return None, None # ou le format attendu pour que le loader puisse gérer l'exception

    # 2. Si des éléments valides restent, on les combine
    return tuple(zip(*valid_batch))


# 5️⃣ EARLY STOPPING (AMÉLIORÉ)
class EarlyStopping:
    """Gestion early stopping avec choix de métrique"""
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

    def __call__(self, current_value, model):
        improved = (self.mode == 'max' and current_value > self.best_value + self.min_delta) or \
                   (self.mode == 'min' and current_value < self.best_value - self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        return model


# 6️⃣ ÉVALUATION avec mAP@50
@torch.inference_mode()
def evaluate(model, val_loader, device, writer, epoch):
    """Validation avec mAP@50"""
    # Temporairement définir le modèle en mode train pour obtenir les sorties de perte
    # mais conserver no_grad pour éviter les mises à jour de poids
    model.train() # Temporarily set to train mode to get loss outputs

    metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy')

    val_loss = 0
    num_batches = 0

    for imgs, targets in tqdm(val_loader, desc="Validation", leave=False):
        if imgs is None or targets is None: # Gérer les lots vides après filtrage
            continue

        imgs = list(img.to(device) for img in imgs)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pour calculer la loss
        loss_dict = model(imgs, targets_device)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
        num_batches += 1

        # Inference pour mAP
        model.eval() # Set to eval mode for inference
        preds = model(imgs)
        model.train() # Set back to train mode for next loss calculation

        # Préparer pour torchmetrics (format COCO)
        preds_formatted = []
        targets_formatted = []

        for pred, target in zip(preds, targets):
            # Prédictions
            preds_formatted.append({
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu(),
            })

            # Targets
            targets_formatted.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
            })

        if preds_formatted and targets_formatted: # Ensure not empty lists
            metric.update(preds_formatted, targets_formatted)

    model.eval() # Set model back to eval mode after loss calculation

    # Calculer mAP
    if num_batches > 0: # Avoid division by zero if no batches processed
        results = metric.compute()
        map50 = results['map_50'].item()
        map_all = results['map'].item()
        avg_val_loss = val_loss / num_batches
    else:
        map50 = 0.0
        map_all = 0.0
        avg_val_loss = 0.0


    # Logging TensorBoard
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    writer.add_scalar('val/mAP50', map50, epoch)
    writer.add_scalar('val/mAP', map_all, epoch)

    # Reset metric
    metric.reset()

    return avg_val_loss, map50, map_all


# 7️⃣ ENTRAÎNEMENT
def train_one_epoch(model, train_loader, optimizer, device, writer, epoch):
    """Une epoch d'entraînement"""
    model.train()
    train_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")

    for images, targets in pbar:
        if images is None or targets is None: # Gérer les lots vides après filtrage
            continue

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

    if num_batches > 0: # Avoid division by zero if no batches processed
        avg_train_loss = train_loss / num_batches
    else:
        avg_train_loss = 0.0

    writer.add_scalar('train/loss', avg_train_loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    return avg_train_loss


# 8️⃣ SCRIPT PRINCIPAL
def main():
    """Boucle d'entraînement complète"""
    global config
    config = get_config()

    # Créer dossiers
    config['save_dir'].mkdir(parents=True, exist_ok=True)
    config['log_dir'].mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # TensorBoard
    writer = SummaryWriter(config['log_dir'])
    print(f"📊 TensorBoard: tensorboard --logdir={config['log_dir']}")

    # Modèle
    num_classes = len(CLASS_NAMES) + 1
    print(f"📦 Classes: {num_classes} (incl. background)")

    model = build_fasterrcnn_light_model(num_classes)
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )

    # Early stopping (sur mAP50 par défaut)
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        metric=config['early_stopping_metric'],
        mode=config['early_stopping_mode']
    )

    # Historique
    history = {'train_loss': [], 'val_loss': [], 'map50': [], 'map': []}

    # Boucle d'entraînement
    print("🚀 Début de l'entraînement...")
    print(f"🛑 Early stopping sur {config['early_stopping_metric']} ({config['early_stopping_mode']})")

    for epoch in range(1, config['num_epochs'] + 1):
        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, device, writer, epoch)

        # Validation avec mAP
        val_loss, map50, map_all = evaluate(model, val_loader, device, writer, epoch)

        # Scheduler
        scheduler.step()

        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['map50'].append(map50)
        history['map'].append(map_all)

        print(f"\n📈 Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")
        print(f"🎯 mAP@50: {map50:.3f} | mAP: {map_all:.3f}")
        print(f"📊 LR: {optimizer.param_groups[0]['lr']:.6f} | ES: {early_stopping.counter}/{early_stopping.patience}")

        # Sauvegarde meilleur modèle (selon la métrique)
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

        # Early stopping
        early_stopping(current_metric, model)
        if early_stopping.should_stop:
            print(f"\n🛑 Early stopping déclenché à l'epoch {epoch}!")
            model = early_stopping.restore_best(model)
            break

    # Sauvegarde finale
    final_path = config['save_dir'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)

    writer.close()
    print(f"\n✅ Entraînement terminé! Modèle final: {final_path}")


if __name__ == '__main__':
    main()