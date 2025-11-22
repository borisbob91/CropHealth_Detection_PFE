#!/usr/bin/env python3
"""
CropHealth Detection - Unified Training Script
Train SSD, EfficientDet, Faster R-CNN (classic & light)

Usage:
    python train.py --model ssd --data data/yolo_crop --device cuda
    python train.py --model efficientdet --data data/coco_crop --device cuda
    python train.py --model fasterrcnn --data data/yolo_crop --device cuda
    python train.py --model fasterrcnn_light --data data/yolo_crop --device cuda
"""
import argparse
import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Imports locaux
from configs.model_configs import MODEL_CONFIGS, NUM_CLASSES
from datasets.yolo_dataset import YoloDataset
from datasets.coco_dataset import CocoDataset
from datasets.pascalvoc_dataset import PascalVOCDataset
from datasets.transforms import get_albu_transform
from models.ssd_model import build_ssd_model
from models.effdet_model import build_efficientdet_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model
from trainers.base_trainer import BaseTrainer


def collate_fn(batch):
    """Custom collate pour detection (images de tailles différentes)"""
    return tuple(zip(*batch))

def build_model(model_key, device):
    """Construit modèle"""
    # Build model
    if model_key == 'ssd':
        model = build_ssd_model(NUM_CLASSES)
        
    elif model_key == 'efficientdet':
        model = build_efficientdet_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn':
        model = build_fasterrcnn_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn_light':
        model = build_fasterrcnn_light_model(NUM_CLASSES)
    elif model_key in ['yolov8n', 'yolov11n']:
        pass
    else:
        raise ValueError(f"Unknown model: {model_key}")
    
    model


def build_dataloaders(model_key, data_root, config):
    """Construit train/val dataloaders selon format dataset"""
    dataset_format = config['dataset_format']
    input_size = config['input_size']
    batch_size = config['batch_size']
    
    if dataset_format == 'yolo':
        # YOLO txt format
        train_imgs = str(Path(data_root) / 'train' / 'images')
        train_lbls = str(Path(data_root) / 'train' / 'labels')
        val_imgs = str(Path(data_root) / 'val' / 'images')
        val_lbls = str(Path(data_root) / 'val' / 'labels')
        test_imgs = str(Path(data_root) / 'test' / 'images')
        test_lbls = str(Path(data_root) / 'test' / 'labels')

        train_ds = YoloDataset(train_imgs, train_lbls, 
                               get_albu_transform(model_key, train=True))
        val_ds = YoloDataset(val_imgs, val_lbls, 
                             get_albu_transform(model_key, train=False))
        test_ds = YoloDataset(test_imgs, test_lbls, 
                              get_albu_transform(model_key, train=False))
    
    elif dataset_format == 'coco':
        # COCO JSON format
        train_json = str(Path(data_root) / 'train' / 'annotations.json')
        train_imgs = str(Path(data_root) / 'train' / 'images')
        val_json = str(Path(data_root) / 'val' / 'annotations.json')
        val_imgs = str(Path(data_root) / 'val' / 'images')
        test_json = str(Path(data_root) / 'test' / 'annotations.json')
        test_imgs = str(Path(data_root) / 'test' / 'images')
        
        train_ds = CocoDataset(train_json, train_imgs, 
                               get_albu_transform(model_key, train=True))
        val_ds = CocoDataset(val_json, val_imgs, 
                             get_albu_transform(model_key, train=False))
        test_ds = CocoDataset(test_json, test_imgs, 
                              get_albu_transform(model_key, train=False))
    
    elif dataset_format == 'pascalvoc':
        # Pascal VOC XML format
        train_imgs = str(Path(data_root) / 'train' / 'images')
        train_anns = str(Path(data_root) / 'train' / 'Annotations')
        val_imgs = str(Path(data_root) / 'val' / 'images')
        val_anns = str(Path(data_root) / 'val' / 'Annotations')
        test_imgs = str(Path(data_root) / 'test' / 'images')
        test_anns = str(Path(data_root) / 'test' / 'Annotations')
        
        # Extraire class names depuis config ou auto-detect
        from configs.model_configs import CLASS_NAMES
        
        train_ds = PascalVOCDataset(train_imgs, train_anns, CLASS_NAMES,
                                     get_albu_transform(model_key, train=True))
        val_ds = PascalVOCDataset(val_imgs, val_anns, CLASS_NAMES,
                                   get_albu_transform(model_key, train=False))
        test_ds = PascalVOCDataset(test_imgs, test_anns, CLASS_NAMES,
                                   get_albu_transform(model_key, train=False))
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True,
                             num_workers=2, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


def build_model_optimizer_scheduler(model_key, config, device):
    """Construit modèle, optimizer, scheduler selon config"""
    # Build model
    if model_key == 'ssd':
        model = build_ssd_model(NUM_CLASSES)
    elif model_key == 'efficientdet':
        model = build_efficientdet_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn':
        model = build_fasterrcnn_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn_light':
        model = build_fasterrcnn_light_model(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_key}")
    
    model.to(device)
    
    # Build optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=config['lr'], 
                                    momentum=0.9, weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config['lr'], 
                                      weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config['lr'], 
                                     weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Build scheduler
    if config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
    else:
        # reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
    
    return model, optimizer, scheduler


def main(args):
    # Récupérer config
    config = MODEL_CONFIGS[args.model]
    
    # Créer dossier de sauvegarde
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    save_dir = f"runs/{config['name']}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    
    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args.model, args.data, config)
    
    # Model + Optimizer + Scheduler
    model, optimizer, scheduler = build_model_optimizer_scheduler(
        args.model, config, device
    )
    
    # Trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        model_name=config['name']
    )
    
    # Train
    trainer.train(config['epochs'])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='CropHealth Detection Training')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['ssd', 'efficientdet', 'fasterrcnn', 'fasterrcnn_light'],
                        help='Model architecture')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset root (yolo or coco format or pascalvoc)')
    parser.add_argument('-d', '--device', type=str, default=device,
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)