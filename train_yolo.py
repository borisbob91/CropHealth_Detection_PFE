#!/usr/bin/env python3
"""
CropHealth Detection - YOLOv8n Training
Utilise Ultralytics pour YOLOv8n (augmentations Mosaic + HSV natives)

Usage:
    python train_yolo.py --data data/yolo_crop/data.yaml --device 0
    python train_yolo.py --data data/yolo_crop/data.yaml --device cpu

Structure attendue data.yaml:
    path: /path/to/data/yolo_crop
    train: train/images
    val: val/images
    nc: 9
    names: ['class1', 'class2', ...]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌ pip install ultralytics required")

from configs.model_configs import MODEL_CONFIGS


def train_yolo(args):
    """Entraînement YOLOv8n"""
    config = MODEL_CONFIGS['yolov8n']
    
    # Timestamp pour run unique
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{config['name']}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Training {config['name']}")
    print(f"Data: {args.data}")
    print(f"Epochs: {config['epochs']} | Batch: {config['batch_size']}")
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Run: runs/{run_name}")

    print(f"{'='*60}\n")
    
    # Charger modèle pré-entraîné
    model = YOLO('yolov8n.pt')
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    # Entraînement
    results = model.train(
        data=args.data,
        epochs=config['epochs'],
        imgsz=config['input_size'],
        batch=config['batch_size'],
        lr0=config['lr'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer'],
        cos_lr=True,  # Cosine scheduler
        project='runs',
        name=run_name,
        device=args.device,
        pretrained=True,
        verbose=True,
        cache=args.cache,
        exist_ok=True,
    )
    
    # Validation finale
    if not args.no_val:
        print("\n>>> Final validation...")
        metrics = model.val()
        print(f"✅ mAP@50-95: {metrics.box.map:.4f}")
        print(f"✅ mAP@50: {metrics.box.map50:.4f}")
    
    save_dir = Path(results.save_dir)
    print(f"\n✅ Training complete!")
    print(f"Model saved: {save_dir}/weights/best.pt")
    print(f"TensorBoard: tensorboard --logdir {save_dir}\n")
    
    return model, save_dir


def main():
    parser = argparse.ArgumentParser(description='CropHealth YOLOv8n Training')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0/1/2/3) or cpu')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images in RAM for faster training')
    parser.add_argument('--no-val', action='store_true',
                        help='Skip final validation')
    
    args = parser.parse_args()
    
    # Vérifier data.yaml existe
    if not Path(args.data).exists():
        sys.exit(f"❌ data.yaml not found: {args.data}")
    
    train_yolo(args)


if __name__ == '__main__':
    main()