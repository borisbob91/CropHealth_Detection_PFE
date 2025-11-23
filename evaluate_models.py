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

from configs.model_configs import MODEL_CONFIGS, NUM_CLASSES
from datasets.yolo_dataset import YoloDataset
from datasets.coco_dataset import CocoDataset
from datasets.pascalvoc_dataset import PascalVOCDataset
from train import build_dataloaders

from datasets.transforms import get_albu_transform
from models.ssd_model import build_ssd_model
from models.effdet_model import build_efficientdet_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model

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
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model


def compute_ap_per_class_torchmetrics(predictions, targets, class_names):
    """
    Calcule AP@50 par classe via torchmetrics.detection.MeanAveragePrecision
    
    Args:
        predictions: list de dicts {'boxes', 'labels', 'scores'}
        targets: list de dicts {'boxes', 'labels'}
        class_names: list des noms de classes
    
    Returns:
        dict {class_name: AP@50}, float mAP@50
    """
    # Initialiser metric avec IoU=0.5
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    
    # Update
    metric.update(predictions, targets)
    
    # Compute
    results = metric.compute()
    
    # Extraire mAP@50 global
    map50_global = results['map_50'].item() * 100
    
    # Extraire AP@50 par classe
    ap_per_class = {}
    if 'map_per_class' in results and results['map_per_class'] is not None:
        ap_values = results['map_per_class'].cpu().numpy()
        for i, class_name in enumerate(class_names):
            if i < len(ap_values):
                ap_per_class[class_name] = float(ap_values[i]) * 100
            else:
                ap_per_class[class_name] = 0.0
    else:
        # Fallback si pas de métriques par classe
        for class_name in class_names:
            ap_per_class[class_name] = map50_global
    
    return ap_per_class, map50_global


def compute_f1_per_class(predictions, targets, class_names, iou_threshold=0.5):
    """
    Calcule F1-Score par classe (micro average pour global)
    
    Args:
        predictions: list de dicts {'boxes', 'labels', 'scores'}
        targets: list de dicts {'boxes', 'labels'}
        class_names: list des noms de classes
        iou_threshold: seuil IoU pour match
    
    Returns:
        dict {class_name: {'precision', 'recall', 'f1'}}, dict global (micro average)
    """
    num_classes = len(class_names)
    
    # Compteurs par classe
    tp_per_class = np.zeros(num_classes)
    fp_per_class = np.zeros(num_classes)
    fn_per_class = np.zeros(num_classes)
    
    # Pour chaque image
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        # Trier prédictions par score décroissant
        sorted_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_idx]
        pred_labels = pred_labels[sorted_idx]
        
        # Track matched targets
        matched_targets = set()
        
        # Pour chaque prédiction
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            # Classe 0-indexed
            pred_class_idx = int(pred_label) - 1
            if pred_class_idx < 0 or pred_class_idx >= num_classes:
                continue
            
            # Chercher meilleur match dans targets
            best_iou = 0
            best_target_idx = -1
            
            for i, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                if i in matched_targets:
                    continue
                
                if int(target_label) != int(pred_label):
                    continue
                
                iou = compute_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = i
            
            # Si match trouvé
            if best_iou >= iou_threshold and best_target_idx != -1:
                tp_per_class[pred_class_idx] += 1
                matched_targets.add(best_target_idx)
            else:
                fp_per_class[pred_class_idx] += 1
        
        # FN = targets non matchés
        for i, target_label in enumerate(target_labels):
            if i not in matched_targets:
                target_class_idx = int(target_label) - 1
                if 0 <= target_class_idx < num_classes:
                    fn_per_class[target_class_idx] += 1
    
    # Calculer métriques par classe
    f1_per_class = {}
    
    for i, class_name in enumerate(class_names):
        tp = tp_per_class[i]
        fp = fp_per_class[i]
        fn = fn_per_class[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_per_class[class_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # Micro average global (somme TP/FP/FN sur toutes classes)
    tp_total = tp_per_class.sum()
    fp_total = fp_per_class.sum()
    fn_total = fn_per_class.sum()
    
    precision_global = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_global = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_global = 2 * precision_global * recall_global / (precision_global + recall_global) if (precision_global + recall_global) > 0 else 0
    
    f1_global_dict = {
        'precision': precision_global * 100,
        'recall': recall_global * 100,
        'f1': f1_global * 100
    }
    
    return f1_per_class, f1_global_dict


def compute_iou(box1, box2):
    """Calcule IoU entre deux boxes format [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_precision_recall_per_class(predictions, targets, class_names, iou_threshold=0.5):
    """
    Calcule courbes Precision-Recall par classe
    
    Returns:
        dict {class_name: {'precision': [], 'recall': [], 'auc': float}}
    """
    num_classes = len(class_names)
    pr_curves = {}
    
    # Grouper par classe
    for class_idx, class_name in enumerate(class_names):
        class_label = class_idx + 1  # 1-indexed
        
        all_scores = []
        all_matches = []
        num_gt = 0
        
        # Pour chaque image
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            
            target_boxes = target['boxes'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()
            
            # Prédictions de cette classe
            class_pred_idx = pred_labels == class_label
            class_pred_boxes = pred_boxes[class_pred_idx]
            class_pred_scores = pred_scores[class_pred_idx]
            
            # Targets de cette classe
            class_target_idx = target_labels == class_label
            class_target_boxes = target_boxes[class_target_idx]
            num_gt += len(class_target_boxes)
            
            # Matcher prédictions
            matched_targets = set()
            
            for pred_box, score in zip(class_pred_boxes, class_pred_scores):
                all_scores.append(score)
                
                # Chercher match
                best_iou = 0
                best_idx = -1
                
                for i, target_box in enumerate(class_target_boxes):
                    if i in matched_targets:
                        continue
                    
                    iou = compute_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_iou >= iou_threshold and best_idx != -1:
                    all_matches.append(1)
                    matched_targets.add(best_idx)
                else:
                    all_matches.append(0)
        
        # Calculer courbe PR
        if len(all_scores) > 0 and num_gt > 0:
            # Trier par score décroissant
            sorted_idx = np.argsort(-np.array(all_scores))
            matches_sorted = np.array(all_matches)[sorted_idx]
            
            # Cumsum TP/FP
            tp_cumsum = np.cumsum(matches_sorted)
            fp_cumsum = np.cumsum(1 - matches_sorted)
            
            recall = tp_cumsum / num_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # AUC
            pr_auc = auc(recall, precision) if len(recall) > 1 else 0.0
            
            pr_curves[class_name] = {
                'recall': recall.tolist(),
                'precision': precision.tolist(),
                'auc': pr_auc
            }
        else:
            pr_curves[class_name] = {
                'recall': [0],
                'precision': [0],
                'auc': 0.0
            }
    
    return pr_curves


def compute_confusion_matrix_detection(predictions, targets, num_classes, iou_threshold=0.5):
    """
    Calcule matrice de confusion pour détection
    Lignes = vraies classes, Colonnes = classes prédites
    """
    cm = np.zeros((num_classes - 1, num_classes - 1), dtype=int)
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy() - 1  # 0-indexed
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy() - 1  # 0-indexed
        
        matched_preds = set()
        
        # Pour chaque target
        for target_box, target_label in zip(target_boxes, target_labels):
            if target_label < 0 or target_label >= num_classes - 1:
                continue
            
            # Chercher meilleure prédiction
            best_iou = 0
            best_pred_idx = -1
            
            for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if i in matched_preds:
                    continue
                
                iou = compute_iou(target_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            # Si match trouvé
            if best_iou >= iou_threshold and best_pred_idx != -1:
                pred_label = int(pred_labels[best_pred_idx])
                if 0 <= pred_label < num_classes - 1:
                    cm[int(target_label), pred_label] += 1
                    matched_preds.add(best_pred_idx)
    
    return cm


def plot_ap_per_class(ap_results, output_dir):
    """Plot AP@50 par classe pour tous les modèles"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    class_names = list(next(iter(ap_results.values())).keys())
    x = np.arange(len(class_names))
    width = 0.15
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (model_name, ap_dict) in enumerate(ap_results.items()):
        ap_values = [ap_dict[cls] for cls in class_names]
        ax.bar(x + i * width, ap_values, width, label=model_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('AP@50 (%)', fontsize=12)
    ax.set_title('AP@50 par classe - Comparaison des modèles', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(ap_results) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ap50_per_class_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ AP@50 per class plot: {output_dir / 'ap50_per_class_comparison.png'}")


def plot_f1_per_class(f1_results, output_dir):
    """Plot F1-Score par classe pour tous les modèles"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    class_names = list(next(iter(f1_results.values())).keys())
    x = np.arange(len(class_names))
    width = 0.15
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (model_name, f1_dict) in enumerate(f1_results.items()):
        f1_values = [f1_dict[cls]['f1'] for cls in class_names]
        ax.bar(x + i * width, f1_values, width, label=model_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('F1-Score par classe - Comparaison des modèles', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(f1_results) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_per_class_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ F1-Score per class plot: {output_dir / 'f1_per_class_comparison.png'}")


def plot_map50_comparison(map_results, output_dir):
    """Plot mAP@50 global pour tous les modèles"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(map_results.keys())
    map_values = list(map_results.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(models, map_values, color=colors[:len(models)])
    
    ax.set_ylabel('mAP@50 (%)', fontsize=12)
    ax.set_title('mAP@50 Global - Comparaison des modèles', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'map50_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ mAP@50 comparison plot: {output_dir / 'map50_comparison.png'}")


def plot_precision_recall_curves(pr_curves_all, output_dir):
    """Plot courbes Precision-Recall par classe"""
    class_names = list(next(iter(pr_curves_all.values())).keys())
    
    for class_name in class_names:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, pr_curves in pr_curves_all.items():
            if class_name in pr_curves:
                recall = pr_curves[class_name]['recall']
                precision = pr_curves[class_name]['precision']
                auc_val = pr_curves[class_name]['auc']
                
                ax.plot(recall, precision, label=f'{model_name} (AUC={auc_val:.3f})', linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {class_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        safe_class_name = class_name.replace('/', '_')
        plt.savefig(output_dir / f'pr_curve_{safe_class_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Precision-Recall curves: {output_dir / 'pr_curve_*.png'}")


def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    """Plot matrice de confusion"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix: {output_dir / f'confusion_matrix_{model_name}.png'}")


def save_ap_per_class_csv(ap_results, output_dir):
    """Sauvegarde AP@50 par classe en CSV"""
    csv_path = output_dir / 'ap50_per_class.csv'
    class_names = list(next(iter(ap_results.values())).keys())
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class'] + list(ap_results.keys()))
        
        for class_name in class_names:
            row = [class_name]
            for model_name in ap_results.keys():
                row.append(f"{ap_results[model_name][class_name]:.2f}")
            writer.writerow(row)
    
    print(f"✅ AP@50 per class CSV: {csv_path}")


def save_f1_per_class_csv(f1_results, output_dir):
    """Sauvegarde F1-Score par classe en CSV"""
    csv_path = output_dir / 'f1_per_class.csv'
    class_names = list(next(iter(f1_results.values())).keys())
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Class', 'Metric']
        for model_name in f1_results.keys():
            header.append(model_name)
        writer.writerow(header)
        
        # Données
        for class_name in class_names:
            # Precision
            row_p = [class_name, 'Precision (%)']
            for model_name in f1_results.keys():
                row_p.append(f"{f1_results[model_name][class_name]['precision']:.2f}")
            writer.writerow(row_p)
            
            # Recall
            row_r = [class_name, 'Recall (%)']
            for model_name in f1_results.keys():
                row_r.append(f"{f1_results[model_name][class_name]['recall']:.2f}")
            writer.writerow(row_r)
            
            # F1
            row_f = [class_name, 'F1-Score (%)']
            for model_name in f1_results.keys():
                row_f.append(f"{f1_results[model_name][class_name]['f1']:.2f}")
            writer.writerow(row_f)
            
            writer.writerow([])  # Ligne vide
    
    print(f"✅ F1-Score per class CSV: {csv_path}")


def save_global_metrics_csv(map_results, f1_global_results, output_dir):
    """Sauvegarde métriques globales (mAP@50 + F1 micro average)"""
    csv_path = output_dir / 'global_metrics.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'mAP@50 (%)', 'Precision (%) - Micro', 'Recall (%) - Micro', 'F1-Score (%) - Micro'])
        
        for model_name in map_results.keys():
            writer.writerow([
                model_name,
                f"{map_results[model_name]:.2f}",
                f"{f1_global_results[model_name]['precision']:.2f}",
                f"{f1_global_results[model_name]['recall']:.2f}",
                f"{f1_global_results[model_name]['f1']:.2f}"
            ])
    
    print(f"✅ Global metrics CSV: {csv_path}")


def evaluate_single_model(model_key, checkpoint_path, val_data, device, class_names):
    """Évalue un seul modèle"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_key.upper()}")
    print(f"{'='*60}")
    
    # Charger modèle
    model = build_model(model_key, checkpoint_path, device)
    config = MODEL_CONFIGS.get(model_key, {'dataset_format': 'yolo'})
    
    train_loader, val_loader, test_loader = build_dataloaders(model_key,Path(val_data) )
    # Inférence
    print("Running inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = list(img.to(device) for img in imgs)
            preds = model(imgs)
            
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            all_preds.extend(preds_cpu)
            all_targets.extend(targets_cpu)
    
    # Calculer métriques
    print("Computing metrics...")
    
    # 1. AP@50 par classe (torchmetrics)
    ap_per_class, map50 = compute_ap_per_class_torchmetrics(all_preds, all_targets, class_names)
    
    # 2. F1-Score par classe + global (micro average)
    f1_per_class, f1_global = compute_f1_per_class(all_preds, all_targets, class_names)
    
    # 3. Precision-Recall curves
    pr_curves = compute_precision_recall_per_class(all_preds, all_targets, class_names)
    
    # 4. Confusion Matrix
    cm = compute_confusion_matrix_detection(all_preds, all_targets, NUM_CLASSES)
    
    print(f"✅ {model_key} - mAP@50: {map50:.2f}% | F1 (micro): {f1_global['f1']:.2f}%")
    
    return {
        'ap_per_class': ap_per_class,
        'map50': map50,
        'f1_per_class': f1_per_class,
        'f1_global': f1_global,
        'pr_curves': pr_curves,
        'confusion_matrix': cm
    }


def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Class names
    class_names = [f'class_{i}' for i in range(NUM_CLASSES - 1)]
    
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Multi-Model Evaluation")
    print(f"Output: {output_dir}")
    print(f"Classes: {class_names}")
    print(f"{'='*60}")
    
    # Évaluer chaque modèle
    all_results = {}
    ap_results = {}
    map_results = {}
    f1_results = {}
    f1_global_results = {}
    pr_curves_all = {}
    
    for model_key, checkpoint in args.checkpoints.items():
        results = evaluate_single_model(model_key, checkpoint, args.val_data, device, class_names)
        
        all_results[model_key] = results
        ap_results[model_key] = results['ap_per_class']
        map_results[model_key] = results['map50']
        f1_results[model_key] = results['f1_per_class']
        f1_global_results[model_key] = results['f1_global']
        pr_curves_all[model_key] = results['pr_curves']
        
        # Plot confusion matrix individuelle
        plot_confusion_matrix(results['confusion_matrix'], class_names, model_key, output_dir)
    
    # Plots comparatifs
    print(f"\n{'='*60}")
    print(f"Generating comparison plots...")
    print(f"{'='*60}\n")
    
    plot_ap_per_class(ap_results, output_dir)
    plot_f1_per_class(f1_results, output_dir)
    plot_map50_comparison(map_results, output_dir)
    plot_precision_recall_curves(pr_curves_all, output_dir)
    
    # Sauvegarder CSV
    save_ap_per_class_csv(ap_results, output_dir)
    save_f1_per_class_csv(f1_results, output_dir)
    save_global_metrics_csv(map_results, f1_global_results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"{'='*60}\n")
    
    # Afficher résumé
    print("📈 SUMMARY:")
    print(f"{'Model':<20} {'mAP@50':<12} {'F1-Score (micro)':<20}")
    print("-" * 52)
    for model_key in map_results.keys():
        print(f"{model_key:<20} {map_results[model_key]:>10.2f}%  {f1_global_results[model_key]['f1']:>17.2f}%")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropHealth Multi-Model Evaluation')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Model checkpoints: model_key:path (e.g., ssd:runs/SSD/best.pt)')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Validation dataset root')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Parse checkpoints
    checkpoints_dict = {}
    for ckpt_str in args.checkpoints:
        model_key, path = ckpt_str.split(':')
        checkpoints_dict[model_key] = path
    
    args.checkpoints = checkpoints_dict
    
    main(args)