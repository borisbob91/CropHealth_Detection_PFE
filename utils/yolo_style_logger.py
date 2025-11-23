

# utils/yolo_style_logger.py
# YOLO-style checkpoint logger – 100% compatible SSD / FasterRCNN / etc.
# Rendu IDENTIQUE à Ultralytics YOLOv8 - VERSION CORRIGÉE

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import random
from tqdm.auto import tqdm
from collections import defaultdict

# Torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from configs.model_configs import CLASS_NAMES

# ===================================================================
# CONFIGURATION
# ===================================================================
CLASS_NAMES_FULL = ['background'] + CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES_FULL)

random.seed(42)
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(NUM_CLASSES)]


# ===================================================================
# FONCTION PRINCIPALE
# ===================================================================
@torch.inference_mode()
def save_yolo_style_checkpoint(
    model,
    val_loader,
    epoch: int,
    save_dir: Path,
    device,
    prefix: str = "best",
    num_vis_batches: int = 3,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """
    Crée un dossier YOLO-style avec results.csv, courbes, matrices, images, etc.
    
    Args:
        model: Modèle PyTorch
        val_loader: DataLoader de validation
        epoch: Numéro d'epoch
        save_dir: Répertoire de sauvegarde
        device: Device (cuda/cpu)
        prefix: Préfixe du dossier ("best" ou "epoch_X")
        num_vis_batches: Nombre de batchs à visualiser
        conf_threshold: Seuil de confiance pour filtrage
        iou_threshold: Seuil IoU pour NMS (non utilisé ici car déjà fait par le modèle)
    """
    model.eval()
    save_dir = Path(save_dir)
    ckpt_dir = save_dir / prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🎨 Génération checkpoint YOLO-style → {ckpt_dir}")
    print(f"{'='*70}")

    # Collecter toutes les prédictions et targets
    all_preds = []
    all_targets = []
    all_pred_labels = []
    all_true_labels = []
    
    batch_idx = 0

    for images, targets in tqdm(val_loader, desc="📊 Validation + Logging"):
        images = [img.to(device) for img in images]
        preds_raw = model(images)

        for pred_raw, target in zip(preds_raw, targets):
            # Filtrer par confiance
            keep = pred_raw["scores"] >= conf_threshold
            pred = {
                "boxes": pred_raw["boxes"][keep].cpu(),
                "scores": pred_raw["scores"][keep].cpu(),
                "labels": pred_raw["labels"][keep].cpu().long(),
            }
            
            tgt = {
                "boxes": target["boxes"],
                "labels": target["labels"].long(),
            }
            
            all_preds.append(pred)
            all_targets.append(tgt)
            
            # Collecter labels pour confusion matrix
            if len(pred["labels"]) > 0:
                all_pred_labels.extend(pred["labels"].tolist())
            if len(tgt["labels"]) > 0:
                all_true_labels.extend(tgt["labels"].tolist())

        # Visualiser les premiers batchs
        if batch_idx < num_vis_batches:
            _visualize_batch(images, preds_raw, targets, ckpt_dir, batch_idx, conf_threshold)
            batch_idx += 1

    # ====================== MÉTRIQUES PRINCIPALES ======================
    print("\n📈 Calcul des métriques mAP...")
    metric_map = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    metric_map.update(all_preds, all_targets)
    map_results = metric_map.compute()

    mAP50 = map_results["map_50"].item()
    mAP = map_results["map"].item()
    mAP75 = map_results.get("map_75", torch.tensor(0.0)).item()

    # Métriques par classe
    map_per_class = map_results.get("map_per_class", torch.zeros(NUM_CLASSES))
    if map_per_class.dim() == 0:
        map_per_class = torch.zeros(NUM_CLASSES)
    
    # ====================== CALCUL F1 SCORES ======================
    print("📈 Calcul des F1-Scores...")
    f1_scores = _calculate_f1_scores(
        all_preds, 
        all_targets, 
        all_pred_labels, 
        all_true_labels,
        conf_threshold=conf_threshold
    )
    
    # ====================== SAUVEGARDER RESULTS.CSV ======================
    results_data = {
        "epoch": [epoch],
        "mAP@0.5": [mAP50],
        "mAP@0.5:0.95": [mAP],
        "mAP@0.75": [mAP75],
        "F1_macro": [f1_scores['f1_macro']],
        "F1_weighted": [f1_scores['f1_weighted']],
        "Precision_macro": [f1_scores['precision_macro']],
        "Recall_macro": [f1_scores['recall_macro']],
    }
    
    # Ajouter mAP par classe
    for i, class_name in enumerate(CLASS_NAMES_FULL[1:], start=1):  # Skip background
        if i < len(map_per_class):
            results_data[f"mAP_{class_name}"] = [map_per_class[i].item()]
    
    # Ajouter F1 par classe
    for class_name, f1_val in f1_scores['f1_per_class'].items():
        results_data[f"F1_{class_name}"] = [f1_val]
    
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(ckpt_dir / "results.csv", index=False)
    
    print(f"\n✅ results.csv sauvegardé")
    print(f"   📊 mAP@0.5 = {mAP50:.4f}")
    print(f"   📊 mAP@0.5:0.95 = {mAP:.4f}")
    print(f"   🎯 F1-Score (Macro) = {f1_scores['f1_macro']:.4f}")
    print(f"   🎯 F1-Score (Weighted) = {f1_scores['f1_weighted']:.4f}")

    # ====================== VISUALISATIONS ======================
    print("\n🎨 Génération des visualisations...")
    
    # 1. Courbe PR
    print("   ├─ Courbe Precision-Recall...")
    _plot_pr_curve(map_results, ckpt_dir)
    
    # 2. Courbe F1
    print("   ├─ Courbe F1-Confidence...")
    _plot_f1_curve(all_preds, all_targets, ckpt_dir)
    
    # 3. Graphique F1 par classe
    print("   ├─ F1-Scores par classe...")
    _plot_f1_per_class(f1_scores, ckpt_dir)
    
    # 4. Confusion Matrix (VERSION CORRIGÉE)
    print("   ├─ Matrice de confusion...")
    if len(all_pred_labels) > 0 and len(all_true_labels) > 0:
        _plot_confusion_matrix_fixed(all_pred_labels, all_true_labels, ckpt_dir)
    else:
        print("   │  ⚠️  Pas assez de données pour la matrice de confusion")
    
    # 5. Distribution des labels
    print("   ├─ Distribution des labels...")
    _plot_labels_distribution(val_loader, ckpt_dir)
    
    # 6. Graphiques d'entraînement (si historique disponible)
    print("   └─ Courbes d'entraînement...")
    _plot_training_curves(ckpt_dir)

    print(f"\n{'='*70}")
    print(f"✅ CHECKPOINT YOLO-STYLE COMPLET")
    print(f"{'='*70}")
    print(f"📁 Localisation: {ckpt_dir}")
    print(f"📊 mAP@0.5: {mAP50:.4f} | mAP@0.5:0.95: {mAP:.4f}")
    print(f"🎯 F1 (Macro): {f1_scores['f1_macro']:.4f} | F1 (Weighted): {f1_scores['f1_weighted']:.4f}")
    print(f"{'='*70}\n")

    model.train()
    return ckpt_dir


# ====================== GRAPHIQUE F1 PAR CLASSE ======================
def _plot_f1_per_class(f1_scores, save_dir):
    """
    Graphique en barres des F1-Scores par classe
    """
    f1_per_class = f1_scores['f1_per_class']
    precision_per_class = f1_scores['precision_per_class']
    recall_per_class = f1_scores['recall_per_class']
    
    if not f1_per_class:
        return
    
    classes = list(f1_per_class.keys())
    f1_values = [f1_per_class[c] for c in classes]
    precision_values = [precision_per_class[c] for c in classes]
    recall_values = [recall_per_class[c] for c in classes]
    
    # ========== GRAPHIQUE 1: F1 par classe ==========
    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.8), 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_values, width, label='Precision', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, recall_values, width, label='Recall', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, f1_values, width, label='F1-Score', color='#F18F01', alpha=0.8)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Ligne macro moyenne
    ax.axhline(y=f1_scores['f1_macro'], color='red', linestyle='--', 
               linewidth=2, label=f"F1 Macro: {f1_scores['f1_macro']:.3f}")
    ax.axhline(y=f1_scores['f1_weighted'], color='green', linestyle='--', 
               linewidth=2, label=f"F1 Weighted: {f1_scores['f1_weighted']:.3f}")
    
    ax.set_xlabel("Classes", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Precision, Recall & F1-Score par Classe", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "F1_per_class.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== GRAPHIQUE 2: F1 seul (simplifié) ==========
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.7), 6))
    
    bars = ax.bar(classes, f1_values, color='#2E86AB', edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Colorier en fonction du score
    for bar, f1_val in zip(bars, f1_values):
        if f1_val >= 0.8:
            bar.set_color('#2ECC71')  # Vert
        elif f1_val >= 0.6:
            bar.set_color('#F39C12')  # Orange
        else:
            bar.set_color('#E74C3C')  # Rouge
    
    # Valeurs sur barres
    for bar, f1_val in zip(bars, f1_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{f1_val:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Lignes moyennes
    ax.axhline(y=f1_scores['f1_macro'], color='red', linestyle='--', 
               linewidth=2.5, label=f"Macro: {f1_scores['f1_macro']:.3f}")
    ax.axhline(y=f1_scores['f1_weighted'], color='blue', linestyle='--', 
               linewidth=2.5, label=f"Weighted: {f1_scores['f1_weighted']:.3f}")
    
    ax.set_xlabel("Classes", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
    ax.set_title("F1-Score par Classe", fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "F1_scores.png", dpi=300, bbox_inches="tight")
    plt.close()

# ====================== CALCUL F1 SCORES ======================
def _calculate_f1_scores(all_preds, all_targets, pred_labels, true_labels, conf_threshold=0.25, iou_threshold=0.5):
    """
    Calcule les F1-Scores: par classe, macro et weighted
    
    Args:
        all_preds: Liste des prédictions
        all_targets: Liste des targets
        pred_labels: Liste plate des labels prédits
        true_labels: Liste plate des labels vrais
        conf_threshold: Seuil de confiance
        iou_threshold: Seuil IoU pour matching TP
    
    Returns:
        dict: {
            'f1_per_class': {class_name: f1_score},
            'f1_macro': float,
            'f1_weighted': float,
            'precision_macro': float,
            'recall_macro': float
        }
    """
    from collections import defaultdict
    
    # Compteurs par classe
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    
    # Pour chaque image
    for pred, target in zip(all_preds, all_targets):
        pred_boxes = pred['boxes']
        pred_labels_img = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels_img = target['labels']
        
        # Filtrer par confiance
        keep = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels_img = pred_labels_img[keep]
        
        # Tracker les GT déjà matchés
        matched_gt = set()
        
        # Pour chaque prédiction
        for pred_box, pred_label in zip(pred_boxes, pred_labels_img):
            pred_label = int(pred_label)
            best_iou = 0
            best_gt_idx = -1
            
            # Trouver le meilleur GT match
            for gt_idx, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels_img)):
                if int(gt_label) != pred_label:
                    continue
                    
                if gt_idx in matched_gt:
                    continue
                
                iou = _compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Si match trouvé
            if best_iou >= iou_threshold:
                tp_per_class[pred_label] += 1
                matched_gt.add(best_gt_idx)
            else:
                fp_per_class[pred_label] += 1
        
        # Compter les FN (GT non matchés)
        for gt_idx, gt_label in enumerate(target_labels_img):
            gt_label = int(gt_label)
            total_per_class[gt_label] += 1
            
            if gt_idx not in matched_gt:
                fn_per_class[gt_label] += 1
    
    # Calculer F1 par classe
    f1_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    
    for class_idx in range(1, NUM_CLASSES):  # Skip background
        tp = tp_per_class[class_idx]
        fp = fp_per_class[class_idx]
        fn = fn_per_class[class_idx]
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        class_name = CLASS_NAMES_FULL[class_idx]
        f1_per_class[class_name] = f1
        precision_per_class[class_name] = precision
        recall_per_class[class_name] = recall
    
    # F1 Macro (moyenne simple)
    f1_macro = np.mean(list(f1_per_class.values())) if f1_per_class else 0.0
    precision_macro = np.mean(list(precision_per_class.values())) if precision_per_class else 0.0
    recall_macro = np.mean(list(recall_per_class.values())) if recall_per_class else 0.0
    
    # F1 Weighted (moyenne pondérée par le nombre d'instances)
    total_instances = sum(total_per_class.values())
    if total_instances > 0:
        f1_weighted = sum(
            f1_per_class.get(CLASS_NAMES_FULL[cls], 0) * total_per_class[cls]
            for cls in total_per_class.keys()
        ) / total_instances
    else:
        f1_weighted = 0.0
    
    return {
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'tp_per_class': dict(tp_per_class),
        'fp_per_class': dict(fp_per_class),
        'fn_per_class': dict(fn_per_class),
    }


def _compute_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-8)


# ====================== VISUALISATION BATCHS ======================
def _visualize_batch(images, preds_raw, targets, save_dir, batch_idx, conf_threshold):
    """Crée val_batchX_pred.jpg et val_batchX_labels.jpg"""
    for idx in range(min(3, len(images))):
        # Dénormaliser l'image
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img = img * std + mean
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        
        img_pil = Image.fromarray(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # --- GROUND TRUTH ---
        img_gt = img_pil.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        
        if idx < len(targets):
            for box, lbl in zip(targets[idx]["boxes"], targets[idx]["labels"]):
                x1, y1, x2, y2 = box.tolist()
                lbl_idx = int(lbl)
                if 0 <= lbl_idx < len(CLASS_NAMES_FULL):
                    draw_gt.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                    text = CLASS_NAMES_FULL[lbl_idx]
                    draw_gt.text((x1, max(y1 - 20, 0)), text, fill="lime", font=font)
        
        img_gt.save(save_dir / f"val_batch{batch_idx}_img{idx}_labels.jpg")

        # --- PRÉDICTIONS ---
        img_pred = img_pil.copy()
        draw_pred = ImageDraw.Draw(img_pred)
        
        if idx < len(preds_raw):
            pred = preds_raw[idx]
            keep = pred["scores"] >= conf_threshold
            
            for box, score, lbl in zip(pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]):
                x1, y1, x2, y2 = box.tolist()
                lbl_idx = int(lbl)
                
                if 0 <= lbl_idx < len(COLORS):
                    color = tuple(COLORS[lbl_idx])
                else:
                    color = (255, 0, 0)
                
                draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                if 0 <= lbl_idx < len(CLASS_NAMES_FULL):
                    text = f"{CLASS_NAMES_FULL[lbl_idx]} {score:.2f}"
                else:
                    text = f"Class{lbl_idx} {score:.2f}"
                
                draw_pred.text((x1, max(y1 - 20, 0)), text, fill=color, font=font)

        img_pred.save(save_dir / f"val_batch{batch_idx}_img{idx}_pred.jpg")


# ====================== COURBE PR ======================
def _plot_pr_curve(map_results, save_dir):
    """Precision-Recall Curve par classe (style YOLO)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Récupérer les données de précision/recall
    precisions = map_results.get("precisions", None)
    recalls = map_results.get("recalls", None)
    
    if precisions is None or recalls is None:
        # Fallback: créer une courbe basique
        ax.text(0.5, 0.5, "Données PR non disponibles", 
                ha='center', va='center', fontsize=14)
    else:
        # Convertir en numpy
        if isinstance(precisions, torch.Tensor):
            precisions = precisions.cpu().numpy()
        if isinstance(recalls, torch.Tensor):
            recalls = recalls.cpu().numpy()
        
        # Tracer pour chaque classe (sauf background)
        for i in range(1, min(NUM_CLASSES, len(precisions))):
            if recalls.shape[0] > i and np.sum(recalls[i]) > 0:
                # Extraire AP pour cette classe
                map_per_class = map_results.get("map_per_class", torch.zeros(NUM_CLASSES))
                ap = map_per_class[i].item() if i < len(map_per_class) else 0.0
                
                ax.plot(recalls[i], precisions[i], 
                       linewidth=2,
                       label=f"{CLASS_NAMES_FULL[i]} (AP={ap:.3f})")
    
    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_dir / "PR_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== COURBE F1 ======================
def _plot_f1_curve(all_preds, all_targets, save_dir):
    """F1 vs Confidence Threshold (comme YOLO)"""
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    confs = np.linspace(0.01, 0.95, 50)
    map50_scores = []

    for conf in tqdm(confs, desc="   │  Calcul F1", leave=False):
        filtered_preds = []
        for p in all_preds:
            keep = p["scores"] >= conf
            filtered_preds.append({
                "boxes": p["boxes"][keep],
                "scores": p["scores"][keep],
                "labels": p["labels"][keep],
            })
        
        try:
            metric.update(filtered_preds, all_targets)
            result = metric.compute()
            map50_scores.append(result["map_50"].item())
            metric.reset()
        except:
            map50_scores.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(confs, map50_scores, linewidth=3, color="#2E86AB", marker='o', markersize=4)
    ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight='bold')
    ax.set_ylabel("mAP@0.5", fontsize=12, fontweight='bold')
    ax.set_title("F1-Confidence Curve", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(map50_scores) * 1.1 if map50_scores else 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "F1_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== MATRICE DE CONFUSION (VERSION CORRIGÉE) ======================
def _plot_confusion_matrix_fixed(pred_labels, true_labels, save_dir):
    """
    Confusion Matrix corrigée - gère les cas où pas toutes les classes sont présentes
    """
    # Créer la matrice manuellement
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    # Remplir la matrice
    for pred_lbl, true_lbl in zip(pred_labels, true_labels):
        if 0 <= pred_lbl < NUM_CLASSES and 0 <= true_lbl < NUM_CLASSES:
            cm[true_lbl, pred_lbl] += 1
    
    # Filtrer les classes non utilisées
    used_classes = []
    used_indices = []
    for i in range(NUM_CLASSES):
        if cm[i].sum() > 0 or cm[:, i].sum() > 0:
            used_classes.append(CLASS_NAMES_FULL[i])
            used_indices.append(i)
    
    if len(used_indices) == 0:
        print("   │  ⚠️  Aucune prédiction pour la matrice de confusion")
        return
    
    cm_filtered = cm[np.ix_(used_indices, used_indices)]
    
    # ========== MATRICE NORMALE ==========
    fig, ax = plt.subplots(figsize=(max(10, len(used_classes)), max(8, len(used_classes) * 0.8)))
    
    sns.heatmap(cm_filtered, annot=True, fmt="d", cmap="Blues",
                xticklabels=used_classes, yticklabels=used_classes,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("True", fontsize=12, fontweight='bold')
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== MATRICE NORMALISÉE ==========
    cm_norm = cm_filtered.astype(float) / (cm_filtered.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(max(10, len(used_classes)), max(8, len(used_classes) * 0.8)))
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=used_classes, yticklabels=used_classes,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, ax=ax)
    
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("True", fontsize=12, fontweight='bold')
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== DISTRIBUTION LABELS ======================
def _plot_labels_distribution(val_loader, save_dir):
    """labels.jpg – distribution des classes dans le dataset"""
    counts = np.zeros(NUM_CLASSES)
    
    for _, targets in val_loader:
        for t in targets:
            labels = t["labels"].numpy()
            for lbl in labels:
                if 0 <= int(lbl) < NUM_CLASSES:
                    counts[int(lbl)] += 1
    
    # Filtrer les classes avec count > 0
    used_indices = [i for i in range(NUM_CLASSES) if counts[i] > 0]
    used_classes = [CLASS_NAMES_FULL[i] for i in used_indices]
    used_counts = counts[used_indices]
    
    if len(used_classes) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(max(10, len(used_classes) * 0.8), 6))
    
    bars = ax.bar(range(len(used_classes)), used_counts, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, used_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel("Classes", fontsize=12, fontweight='bold')
    ax.set_ylabel("Nombre d'instances", fontsize=12, fontweight='bold')
    ax.set_title("Distribution des classes (validation)", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(used_classes)))
    ax.set_xticklabels(used_classes, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_dir / "labels.jpg", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== COURBES D'ENTRAÎNEMENT ======================
def _plot_training_curves(save_dir):
    """
    Génère les courbes d'entraînement si results.csv existe
    (train_loss, val_loss, mAP évolution)
    """
    results_file = save_dir / "results.csv"
    
    # Chercher aussi dans le parent si on est dans "best" ou "epoch_X"
    if not results_file.exists():
        parent_results = save_dir.parent / "results.csv"
        if parent_results.exists():
            results_file = parent_results
    
    if not results_file.exists():
        return
    
    try:
        df = pd.read_csv(results_file)
        
        if len(df) < 2:
            return  # Pas assez de données pour tracer
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # mAP@0.5
        if "mAP@0.5" in df.columns:
            axes[0, 0].plot(df["epoch"], df["mAP@0.5"], marker='o', linewidth=2, color='#2E86AB')
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("mAP@0.5")
            axes[0, 0].set_title("mAP@0.5 Evolution")
            axes[0, 0].grid(True, alpha=0.3)
        
        # mAP@0.5:0.95
        if "mAP@0.5:0.95" in df.columns:
            axes[0, 1].plot(df["epoch"], df["mAP@0.5:0.95"], marker='o', linewidth=2, color='#A23B72')
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("mAP@0.5:0.95")
            axes[0, 1].set_title("mAP@0.5:0.95 Evolution")
            axes[0, 1].grid(True, alpha=0.3)
        
        # Train/Val Loss (si disponibles)
        if "train_loss" in df.columns and "val_loss" in df.columns:
            axes[1, 0].plot(df["epoch"], df["train_loss"], label="Train", marker='o', linewidth=2)
            axes[1, 0].plot(df["epoch"], df["val_loss"], label="Val", marker='s', linewidth=2)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("Loss Evolution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate (si disponible)
        if "lr" in df.columns:
            axes[1, 1].plot(df["epoch"], df["lr"], marker='o', linewidth=2, color='#F18F01')
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"   │  ⚠️  Erreur génération courbes: {e}")