# utils/yolo_style_logger.py
# YOLO-style checkpoint logger – 100 % compatible SSD / FasterRCNN / etc.
# Rendu IDENTIQUE à Ultralytics YOLOv8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import random
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm

# Torchmetrics (imports stables et existants)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from torchmetrics.detection import ConfusionMatrix

# ===================================================================
# À ADAPTER UNE SEULE FOIS (tes vraies classes)
# ===================================================================
CLASS_NAMES = ['background', 'healthy', 'diseased', 'pest']  # ← Remplace par tes CLASS_NAMES
random.seed(42)  # Pour couleurs fixes
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS_NAMES))]

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
    prefix: str = "best",           # "best" ou f"epoch_{epoch}"
    num_vis_batches: int = 3,
    conf_threshold: float = 0.25,
):
    """
    Crée un dossier YOLO-style avec results.csv, courbes, matrices, images, etc.
    """
    model.eval()
    save_dir = Path(save_dir)
    ckpt_dir = save_dir / prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Generating YOLO-style checkpoint → {ckpt_dir}")

    all_preds = []
    all_targets = []

    metric_map = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    metric_cm = ConfusionMatrix(task="multiclass", num_classes=len(CLASS_NAMES))

    batch_idx = 0

    for images, targets in tqdm(val_loader, desc="Validation + Logging"):
        images = [img.to(device) for img in images]
        preds_raw = model(images)

        for pred_raw, target in zip(preds_raw, targets):
            # Filtrer par confiance
            keep = pred_raw["scores"] >= conf_threshold
            pred = {
                "boxes": pred_raw["boxes"][keep].cpu(),
                "scores": pred_raw["scores"][keep].cpu(),
                "labels": pred_raw["labels"][keep].cpu().to(torch.long),
            }
            all_preds.append(pred)

            tgt = {
                "boxes": target["boxes"],
                "labels": target["labels"].to(torch.long),
            }
            all_targets.append(tgt)

            # Update confusion matrix
            if len(pred["labels"]) > 0:
                metric_cm.update(pred["labels"], tgt["labels"])

        # Visualiser les premiers batchs
        if batch_idx < num_vis_batches:
            _visualize_batch(images, preds_raw, targets, ckpt_dir, batch_idx)
            batch_idx += 1

    # ====================== MÉTRIQUES PRINCIPALES ======================
    metric_map.update(all_preds, all_targets)
    map_results = metric_map.compute()

    mAP50 = map_results["map_50"].item()
    mAP = map_results["map"].item()

    # results.csv (format exact YOLO)
    pd.DataFrame({
        "epoch": [epoch],
        "mAP@0.5": [mAP50],
        "mAP@0.5:0.95": [mAP],
    }).to_csv(ckpt_dir / "results.csv", index=False)

    # ====================== VISUALISATIONS ======================
    _plot_pr_curve(map_results, ckpt_dir)
    _plot_f1_curve(all_preds, all_targets, ckpt_dir)

    # Confusion Matrix
    cm = metric_cm.compute().cpu().numpy()
    _plot_confusion_matrix(cm, ckpt_dir, normalized=False)
    _plot_confusion_matrix(cm, ckpt_dir, normalized=True)

    # Distribution des labels
    _plot_labels_distribution(val_loader, ckpt_dir)

    print(f"✅ YOLO-style checkpoint COMPLET → {ckpt_dir}")
    print(f"   mAP@0.5 = {mAP50:.4f} | mAP@0.5:0.95 = {mAP:.4f}")

    model.train()
    return ckpt_dir


# ====================== VISUALISATION BATCHS ======================
def _visualize_batch(images, preds_raw, targets, save_dir, batch_idx):
    """Crée val_batchX_pred.jpg et val_batchX_labels.jpg"""
    for idx in range(min(3, len(images))):  # Max 3 images par batch
        # Dénormaliser l'image
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img = img * std + mean
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # --- GROUND TRUTH ---
        img_gt = img_pil.copy()
        draw_gt = ImageDraw.Draw(img_gt)
        for box, lbl in zip(targets[idx]["boxes"], targets[idx]["labels"]):
            x1, y1, x2, y2 = box.tolist()
            draw_gt.rectangle([x1, y1, x2, y2], outline="lime", width=4)
            draw_gt.text((x1, max(y1 - 30, 0)), CLASS_NAMES[int(lbl)], fill="lime", font=font)
        
        img_gt.save(save_dir / f"val_batch{batch_idx}_labels.jpg")

        # --- PRÉDICTIONS ---
        pred = preds_raw[idx]
        keep = pred["scores"] >= 0.25
        for box, score, lbl in zip(pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]):
            x1, y1, x2, y2 = box.tolist()
            color = tuple(COLORS[int(lbl) % len(COLORS)])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{CLASS_NAMES[int(lbl)]} {score:.2f}"
            draw.text((x1, max(y1 - 30, 0)), text, fill=color, font=font)

        img_pil.save(save_dir / f"val_batch{batch_idx}_pred.jpg")


# ====================== COURBE PR ======================
def _plot_pr_curve(map_results, save_dir):
    """Precision-Recall Curve par classe (comme YOLO)"""
    plt.figure(figsize=(10, 8))
    
    # Extraire precisions/recalls de torchmetrics (disponibles dans map_results)
    precisions = map_results.get("precisions", torch.zeros((len(CLASS_NAMES), 101))).numpy()
    recalls = map_results.get("recalls", torch.zeros((len(CLASS_NAMES), 101))).numpy()
    
    for i in range(1, len(CLASS_NAMES)):  # Skip background
        if np.sum(recalls[i]) > 0:  # Éviter courbes vides
            plt.plot(recalls[i], precisions[i], 
                     label=f"{CLASS_NAMES[i]} (AP={map_results['map_per_class'][i]:.3f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_dir / "PR_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== COURBE F1 ======================
def _plot_f1_curve(all_preds, all_targets, save_dir):
    """F1 vs Confidence Threshold (la fameuse courbe YOLO)"""
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    confs = np.linspace(0.01, 0.99, 80)
    scores = []

    for conf in tqdm(confs, desc="F1 curve", leave=False):
        filtered_preds = []
        for p in all_preds:
            keep = p["scores"] >= conf
            filtered_preds.append({
                "boxes": p["boxes"][keep],
                "scores": p["scores"][keep],
                "labels": p["labels"][keep],
            })
        
        metric.update(filtered_preds, all_targets)
        scores.append(metric.compute()["map_50"].item())
        metric.reset()

    plt.figure(figsize=(9, 6))
    plt.plot(confs, scores, linewidth=3, color="#2E86AB")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("mAP@0.5 (proxy F1)")
    plt.title("F1-Confidence Curve")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dir / "F1_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== MATRICE DE CONFUSION ======================
def _plot_confusion_matrix(cm, save_dir, normalized=False):
    """Confusion Matrix (normale + normalisée)"""
    plt.figure(figsize=(10, 8))
    if normalized:
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Confusion Matrix (normalized)")
        plt.savefig(save_dir / "confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Confusion Matrix")
        plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================== DISTRIBUTION LABELS ======================
def _plot_labels_distribution(val_loader, save_dir):
    """labels.jpg – distribution des classes"""
    counts = np.zeros(len(CLASS_NAMES))
    for _, targets in val_loader:
        for t in targets:
            labels = t["labels"].numpy()
            u, c = np.unique(labels, return_counts=True)
            for uu, cc in zip(u, c):
                counts[int(uu)] += cc

    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASS_NAMES, y=counts, palette="viridis")
    plt.title("Distribution des classes (validation)")
    plt.ylabel("Nombre d'instances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "labels.jpg", dpi=300)
    plt.close()