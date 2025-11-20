#!/usr/bin/env python3
"""
CropHealth Detection - Inference Script
Prédictions + visualisations + métriques pour SSD, EfficientDet, Faster R-CNN
Utilise COCO Evaluator (PyTorch TorchVision) selon protocole standard

Usage:
    python predict.py --model ssd --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
                      --input data/test/images --output predictions/ssd \
                      --val-data data/yolo_crop
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.pascalvoc_dataset import PascalVOCDataset
from thop import profile
from torchinfo import summary

from .configs.model_configs import MODEL_CONFIGS, NUM_CLASSES
from .datasets.yolo_dataset import YoloDataset
from .datasets.coco_dataset import CocoDataset
from .datasets.transforms import get_albu_transform
from .models.ssd_model import build_ssd_model
from .models.effdet_model import build_efficientdet_model
from .models.frcnn_model import build_fasterrcnn_model
from .models.frcnn_light_model import build_fasterrcnn_light_model


def build_model(model_key, checkpoint_path, device):
    """Charge le modèle depuis checkpoint"""
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
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model


def convert_to_coco_format(dataset, predictions, output_json):
    """Convertit prédictions en format COCO JSON pour COCOeval"""
    coco_gt = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': f'class_{i}'} for i in range(1, NUM_CLASSES)]
    }
    
    coco_pred = []
    ann_id = 1
    
    for img_id, (img, target) in enumerate(dataset):
        # Ground truth
        h, w = img.shape[1], img.shape[2]
        coco_gt['images'].append({'id': img_id, 'width': w, 'height': h})
        
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.tolist()
            coco_gt['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1),
                'iscrowd': 0
            })
            ann_id += 1
        
        # Prédictions
        pred = predictions[img_id]
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x1, y1, x2, y2 = box.tolist()
            coco_pred.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score)
            })
    
    # Sauvegarder JSON temporaires
    gt_path = output_json.parent / 'coco_gt.json'
    pred_path = output_json.parent / 'coco_pred.json'
    
    with open(gt_path, 'w') as f:
        json.dump(coco_gt, f)
    with open(pred_path, 'w') as f:
        json.dump(coco_pred, f)
    
    return gt_path, pred_path


def compute_metrics_coco_evaluator(model, dataloader, device, output_dir):
    """
    Calcule métriques via COCO Evaluator (standard PyTorch TorchVision)
    Conforme à Padilla et al. (2020) et protocoles COCO
    """
    print(f"\n📊 Computing metrics via COCO Evaluator...")
    
    all_preds = []
    dataset = dataloader.dataset
    
    # Inférence
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = list(img.to(device) for img in imgs)
            preds = model(imgs)
            
            # CPU
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            all_preds.extend(preds_cpu)
    
    # Convertir en format COCO
    output_json = Path(output_dir) / 'coco_results.json'
    gt_path, pred_path = convert_to_coco_format(dataset, all_preds, output_json)
    
    # COCO Evaluator
    coco_gt = COCO(str(gt_path))
    coco_pred = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extraire métriques
    stats = coco_eval.stats
    metrics = {
        'map': stats[0] * 100,           # mAP@[0.5:0.95]
        'map50': stats[1] * 100,         # mAP@0.5
        'map75': stats[2] * 100,         # mAP@0.75
        'map_small': stats[3] * 100,     # mAP small objects
        'map_medium': stats[4] * 100,    # mAP medium objects
        'map_large': stats[5] * 100,     # mAP large objects
        'precision': stats[1] * 100,     # Approximation via mAP@50
        'recall': stats[8] * 100,        # AR@100
    }
    
    # F1-Score
    p = metrics['precision'] / 100
    r = metrics['recall'] / 100
    metrics['f1'] = (2 * p * r / (p + r) * 100) if (p + r) > 0 else 0
    
    return metrics


def compute_fps(model, input_size, device, num_runs=100):
    """Mesure FPS moyen"""
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model([dummy_input])
    
    # Mesure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model([dummy_input])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    return fps


def compute_model_size_gflops(model, input_size):
    """Calcule taille (MB) et GFLOPs"""
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # GFLOPs via thop
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = macs / 1e9
    
    # Taille via torchinfo
    model_stats = summary(model, input_size=(1, 3, input_size, input_size), verbose=0)
    size_mb = model_stats.total_params * 4 / (1024 ** 2)
    
    return size_mb, gflops


def visualize_predictions(img_path, pred, output_dir, conf_threshold=0.5):
    """Dessine boxes + labels sur image"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        text = f"Class {label}: {score:.2f}"
        draw.text((x1, y1 - 10), text, fill='red')
    
    output_path = output_dir / img_path.name
    img.save(output_path)


def predict_and_save(model, input_dir, output_dir, model_key, device, conf_threshold=0.5):
    """Prédictions sur dossier d'images + sauvegarde visualisations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_paths = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))
    
    print(f"\n🔍 Running predictions on {len(img_paths)} images...")
    
    transform = get_albu_transform(model_key, train=False)
    
    with torch.no_grad():
        for img_path in img_paths:
            img = np.array(Image.open(img_path).convert('RGB'))
            transformed = transform(image=img, bboxes=[], class_labels=[])
            img_t = transformed['image'].unsqueeze(0).to(device)
            
            pred = model([img_t])[0]
            visualize_predictions(img_path, pred, output_dir, conf_threshold)
    
    print(f"✅ Predictions saved to: {output_dir}")


def save_metrics_csv(metrics, output_path):
    """Sauvegarde métriques en CSV selon tableau rapport"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Métrique', 'Rôle', 'Valeur', 'Unité', 'Outil'])
        writer.writerow(['mAP@50', 'Principal', f"{metrics['map50']:.2f}", '%', 'COCO Evaluator'])
        writer.writerow(['mAP@50-95', 'Secondaire', f"{metrics['map']:.2f}", '%', 'COCO Evaluator'])
        writer.writerow(['Precision', 'Analyse FP/FN', f"{metrics['precision']:.2f}", '%', 'COCO Evaluator'])
        writer.writerow(['Recall', 'Analyse FP/FN', f"{metrics['recall']:.2f}", '%', 'COCO Evaluator'])
        writer.writerow(['F1-Score', 'Analyse FP/FN', f"{metrics['f1']:.2f}", '%', 'Calculé (2PR/(P+R))'])
        writer.writerow(['FPS', 'Temps-réel', f"{metrics['fps']:.2f}", 'img/s', 'time.perf_counter()'])
        writer.writerow(['Taille', 'Edge', f"{metrics['size_mb']:.2f}", 'MB', 'torchinfo'])
        writer.writerow(['GFLOPs', 'Edge', f"{metrics['gflops']:.2f}", 'GFLOP', 'thop'])
    
    print(f"✅ Metrics saved to: {output_path}")


def main(args):
    device = torch.device(args.device)
    config = MODEL_CONFIGS[args.model]
    
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Inference")
    print(f"Model: {config['name']}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluator: COCO Evaluator (PyTorch TorchVision)")
    print(f"{'='*60}\n")
    
    # Charger modèle
    model = build_model(args.model, args.checkpoint, device)
    
    # 1. Prédictions + visualisations
    if args.input:
        predict_and_save(model, args.input, args.output, args.model, device, args.conf)
    
    # 2. Métriques sur validation set
    if args.val_data:
        # DataLoader
        if config['dataset_format'] == 'yolo':
            val_imgs = str(Path(args.val_data) / 'val' / 'images')
            val_lbls = str(Path(args.val_data) / 'val' / 'labels')
            val_ds = YoloDataset(val_imgs, val_lbls, get_albu_transform(args.model, train=False))
        elif config['dataset_format'] == 'pascalvoc':
            val_imgs = str(Path(args.val_data) / 'val' / 'images')
            val_ann = str(Path(args.val_data) / 'val' / 'Annotations')
            val_ds = PascalVOCDataset(val_ann, val_imgs, get_albu_transform(args.model, train=False))
        else:
            val_json = str(Path(args.val_data) / 'val' / 'annotations.json')
            val_imgs = str(Path(args.val_data) / 'val' / 'images')
            val_ds = CocoDataset(val_json, val_imgs, get_albu_transform(args.model, train=False))
        
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, 
                                num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
        
        # Métriques COCO Evaluator
        metrics = compute_metrics_coco_evaluator(model, val_loader, device, args.output)
        
        # FPS
        fps = compute_fps(model, config['input_size'], device)
        metrics['fps'] = fps
        
        # Taille + GFLOPs
        size_mb, gflops = compute_model_size_gflops(model, config['input_size'])
        metrics['size_mb'] = size_mb
        metrics['gflops'] = gflops
        # calculer paramètres du modèle
        total_params = sum(p.numel() for p in model.parameters())
        metrics['total_params'] = total_params
        
        # Afficher
        print(f"\n📈 Results (COCO Evaluator):")
        print(f"  mAP@50:       {metrics['map50']:.2f}%")
        print(f"  mAP@50-95:    {metrics['map']:.2f}%")
        print(f"  mAP@75:       {metrics['map75']:.2f}%")
        print(f"  Precision:    {metrics['precision']:.2f}%")
        print(f"  Recall:       {metrics['recall']:.2f}%")
        print(f"  F1-Score:     {metrics['f1']:.2f}%")
        print(f"  FPS:          {metrics['fps']:.2f} img/s")
        print(f"  Size:         {metrics['size_mb']:.2f} MB")
        print(f"  GFLOPs:       {metrics['gflops']:.2f}")
        print(f"  Total Params: {metrics['total_params']:,}")
        
        # Sauvegarder CSV
        csv_path = Path(args.output) / f"{config['name']}_metrics.csv"
        save_metrics_csv(metrics, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropHealth Inference (COCO Evaluator)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['ssd', 'efficientdet', 'fasterrcnn', 'fasterrcnn_light'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt')
    parser.add_argument('--input', type=str,
                        help='Input images directory for predictions')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--val-data', type=str,
                        help='Validation dataset root (for COCO metrics)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for visualization')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)