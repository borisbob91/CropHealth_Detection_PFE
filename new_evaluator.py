#!/usr/bin/env python3
"""
CropHealth Detection - Multi-Model Evaluation
Calcule AP@50 par classe, mAP@50, F1-Score par classe, Precision-Recall curves, Confusion Matrix
Utilise torchmetrics.detection.MeanAveragePrecision pour métriques officielles

Usage:
    python evaluate_models.py --checkpoints ssd:runs/SSD/best.pt fasterrcnn:runs/FRCNN/best.pt \
                              --val-data data/yolo_crop --output evaluation_results/
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve, auc

from configs.model_configs import CLASS_NAMES, MODEL_CONFIGS, NUM_CLASSES
from datasets.yolo_dataset import YoloDataset
from datasets.coco_dataset import CocoDataset
from datasets.pascalvoc_dataset import PascalVOCDataset
from metric_logger import AdvancedYoloLogger
from train import build_dataloaders

from datasets.transforms import get_albu_transform
from models.ssd_model import build_ssd_model
from models.effdet_model import build_efficientdet_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model
from utils.yolo_style_logger import save_yolo_style_checkpoint

def build_model(model_key, checkpoint_path, device):
    """Charge le modèle depuis checkpoint"""
    # Support YOLOv8n via wrapper
    if model_key == 'yolov8n':
        from utils.yolo_wrapper import YOLOv8Wrapper
        model = YOLOv8Wrapper(checkpoint_path)
        model.to(device)
        model.eval()
        return model
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_single_model(model_key, checkpoint_path, val_data, device, class_names=CLASS_NAMES):
    """Évalue un seul modèle"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_key.upper()}")
    print(f"{'='*60}")
    save_dir = Path(checkpoint_path).parent / "evaluation_local_3"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Charger modèle
    model = build_model(model_key, checkpoint_path, device)
    config = MODEL_CONFIGS.get(model_key, {'dataset_format': 'yolo'})
    
    train_loader, val_loader, test_loader = build_dataloaders(model_key, Path(val_data), config )
   
    print(f"🖥️  Device: {device}")
    vis_class_names = ['background'] + class_names
    logger = AdvancedYoloLogger(
        save_dir=save_dir, 
        class_names=vis_class_names,
        device=device
    )

    save_yolo_style_checkpoint(
    model=model,
    val_loader=test_loader,
    epoch=1,
    save_dir=save_dir,
    device=device,
    prefix="best"  # ou f"epoch_{epoch}"
    )
    logger.generate_full_report(model, val_loader, epoch="FINAL")

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = r'C:\Users\BorisBob\Desktop\detection\models\yolov8n_orignal_and_augmented_train\CropHealth_YOLOv8n_1123_1009\weights\best.pt'
    data_root = r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\cotton_crop_dataset_ac_augmented\cotton_crop_yolo_augmented_dataset'
    evaluate_single_model('yolov8n', checkpoint_path, data_root, device)