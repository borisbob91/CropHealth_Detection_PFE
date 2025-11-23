import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

class AdvancedYoloLogger:
    def __init__(self, save_dir, class_names, device):
        """
        Logger complet style YOLO pour GPU puissants.
        Gère : CSV, Courbes Loss, PR Curve, F1 Curve, Confusion Matrix, Visuals.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Ajout explicite de "background" pour la matrice de confusion si pas présent
        self.cm_class_names = class_names + ['background'] 
        self.device = device
        self.csv_path = self.save_dir / 'results.csv'
        self.history = []
        # --- LIGNE DE CORRECTION CRUCIALE ---
        # S'assure que la liste interne du logger a 'background' à l'index 0
        if class_names and class_names[0] != 'background':
            self.class_names = ['background'] + class_names
        else:
            self.class_names = class_names
        
        # Couleurs fixes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8).tolist()

    def log_epoch(self, epoch, metrics_dict):
        """Log rapide à chaque epoch (CSV + Courbes Loss/mAP basiques)"""
        metrics_dict['epoch'] = epoch
        self.history.append(metrics_dict)
        df = pd.DataFrame(self.history)
        
        # Sauvegarde CSV
        cols = ['epoch'] + [c for c in df.columns if c != 'epoch']
        df[cols].to_csv(self.csv_path, index=False)
        
        # Sauvegarde Courbes Loss/mAP temps réel
        self._plot_training_curves(df)

    def generate_full_report(self, model, val_loader, epoch, conf_threshold=0.25, iou_threshold=0.5):
        """
        Génère TOUT le rapport style YOLO (Lourd, à faire à la fin ou sur les checkpoints).
        """
        print(f"📊 Génération du rapport complet style YOLO (Epoch {epoch})...")
        save_folder = self.save_dir / f"checkpoint_epoch_{epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        model.eval()
        all_preds = []
        all_targets = []
        
        # 1. Inference complète sur le val_loader
        print("   ↳ Inference en cours...")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluation")):
                images = [img.to(self.device) for img in images]
                outputs = model(images)

                # Sauvegarder les prédictions et targets pour calculs globaux
                for i, output in enumerate(outputs):
                    # Filtrer par confiance pour alléger
                    mask = output['scores'] > 0.05 
                    all_preds.append({
                        'boxes': output['boxes'][mask].cpu(),
                        'scores': output['scores'][mask].cpu(),
                        'labels': output['labels'][mask].cpu()
                    })
                    all_targets.append({
                        'boxes': targets[i]['boxes'].cpu(),
                        'labels': targets[i]['labels'].cpu()
                    })

                # Visualisation des 3 premiers batchs
                if batch_idx < 3:
                    self._save_visual_batch(images, outputs, targets, save_folder, batch_idx, conf_threshold)

        # 2. Calcul mAP et Courbes PR (via TorchMetrics)
        print("   ↳ Calcul mAP et PR Curve...")
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        metric.update(all_preds, all_targets)
        results = metric.compute()
        
        # Plot PR Curve
        self._plot_pr_curve(results, save_folder)

        # 3. Calcul F1-Confidence Curve
        print("   ↳ Calcul F1 Curve...")
        self._plot_f1_curve(all_preds, all_targets, save_folder)

        # 4. Matrice de Confusion (Calcul "Custom" basé sur l'IoU pour la détection)
        print("   ↳ Calcul Matrice de Confusion...")
        self._compute_and_plot_confusion_matrix(all_preds, all_targets, save_folder, conf_threshold, iou_threshold)

        # 5. Distribution des labels
        print("   ↳ Distribution des classes...")
        self._plot_label_distribution(all_targets, save_folder)
        
        print(f"✅ Rapport complet sauvegardé dans : {save_folder}")

    # ================= INTERNES =================

    def _plot_training_curves(self, df):
        """Courbes Loss/mAP simples mises à jour à chaque epoch"""
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            axs[0].plot(df['epoch'], df['train_loss'], label='Train')
            axs[0].plot(df['epoch'], df['val_loss'], label='Val')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[0].grid(True)

        if 'map50' in df.columns:
            axs[1].plot(df['epoch'], df['map50'], color='orange')
            axs[1].set_title('mAP@50')
            axs[1].grid(True)
            
        if 'map' in df.columns:
            axs[2].plot(df['epoch'], df['map'], color='green')
            axs[2].set_title('mAP@0.5:0.95')
            axs[2].grid(True)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'results.png')
        plt.close()

    def _save_visual_batch(self, images, preds, targets, save_folder, batch_idx, conf_thresh):
        """Génère labels.jpg et preds.jpg pour un batch"""
        vis_labels = []
        vis_preds = []
        
        for i, img_tensor in enumerate(images):
            # Dénormalisation
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1) * 255
            img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

            img_lbl = img_bgr.copy()
            img_pred = img_bgr.copy()

            # GT
            tgt = targets[i]
            for box, lbl in zip(tgt['boxes'].cpu(), tgt['labels'].cpu()):
                self._draw_box(img_lbl, box, int(lbl), color=(0, 255, 0)) # Vert

            # Pred
            p = preds[i]
            keep = p['scores'] > conf_thresh
            for box, lbl, scr in zip(p['boxes'][keep].cpu(), p['labels'][keep].cpu(), p['scores'][keep].cpu()):
                self._draw_box(img_pred, box, int(lbl), score=float(scr))

            vis_labels.append(img_lbl)
            vis_preds.append(img_pred)

        # Grille horizontale
        if vis_labels:
            grid_lbl = np.hstack(vis_labels)
            grid_pred = np.hstack(vis_preds)
            cv2.imwrite(str(save_folder / f'batch_{batch_idx}_labels.jpg'), grid_lbl)
            cv2.imwrite(str(save_folder / f'batch_{batch_idx}_pred.jpg'), grid_pred)

    def _draw_box(self, img, box, label_idx, score=None, color=None):
        x1, y1, x2, y2 = map(int, box)
        if color is None:
            color = tuple(map(int, self.colors[label_idx % len(self.colors)]))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = self.class_names[label_idx] if label_idx < len(self.class_names) else str(label_idx)
        if score: text += f" {score:.2f}"
        
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def _plot_pr_curve(self, results, save_folder):
        """Trace la courbe Precision-Recall par classe"""
        plt.figure(figsize=(10, 8))
        
        # map_per_class est parfois tensor, parfois list selon version torchmetrics
        precisions = results['map_per_class'] 
        
        # Note: Torchmetrics ne renvoie pas directement les courbes brutes facilement
        # On utilise une approximation via les valeurs scalaires ou on skip si complexe
        # ICI: On va faire simple : Afficher les AP par classe en bar plot car extraire la courbe exacte est complexe
        # Si tu veux VRAIMENT la courbe ligne, il faut extraire 'precision' du dict qui est de taille (TxRxK)
        
        # Alternative robuste : Bar plot des AP par classe
        aps = results['map_per_class']
        if isinstance(aps, torch.Tensor): aps = aps.cpu().numpy()
        
        # On ne plotte que si on a des classes valides
        valid_indices = range(len(aps))
        # Ajuster si aps ne matche pas class_names (background exclu ou inclu)
        names = self.class_names[:len(aps)]
        
        sns.barplot(x=names, y=aps, palette="viridis")
        plt.title("mAP per Class")
        plt.ylabel("Average Precision")
        plt.savefig(save_folder / "mAP_per_class.png")
        plt.close()

    def _plot_f1_curve(self, preds, targets, save_folder):
        """Estime la courbe F1 vs Confidence"""
        thresholds = np.linspace(0.1, 0.9, 10)
        f1_scores = []
        
        # C'est une estimation car faire tous les points est trop lent même sur GPU
        for thresh in thresholds:
            # On simule un calcul TP/FP/FN global
            tp, fp, fn = 0, 0, 0
            for p, t in zip(preds, targets):
                p_boxes = p['boxes'][p['scores'] > thresh]
                t_boxes = t['boxes']
                
                if len(p_boxes) == 0:
                    fn += len(t_boxes)
                    continue
                if len(t_boxes) == 0:
                    fp += len(p_boxes)
                    continue
                    
                ious = box_iou(p_boxes, t_boxes)
                # Matching simple (max IoU)
                if ious.numel() > 0:
                    max_ious, _ = ious.max(dim=1)
                    matched = (max_ious > 0.5).sum().item()
                    tp += matched
                    fp += len(p_boxes) - matched
                    fn += len(t_boxes) - matched
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_scores.append(f1)
            
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores, marker='o', linewidth=2)
        plt.title("F1-Score Curve")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.savefig(save_folder / "F1_curve.png")
        plt.close()

    def _compute_and_plot_confusion_matrix(self, preds, targets, save_folder, conf_thresh, iou_thresh):
        """
        Matrice de confusion VÉRITABLE pour détection d'objet.
        Axes: X=Predicted, Y=True. Dernière ligne/col = Background.
        """
        n_classes = len(self.class_names)
        # Matrice (N+1) x (N+1) pour inclure le background
        cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
        
        for p, t in zip(preds, targets):
            p_boxes = p['boxes']
            p_labels = p['labels']
            p_scores = p['scores']
            
            # Filtre confiance
            keep = p_scores >= conf_thresh
            p_boxes = p_boxes[keep]
            p_labels = p_labels[keep]
            
            t_boxes = t['boxes']
            t_labels = t['labels']
            
            if len(p_boxes) == 0:
                # Tout ce qui est target est manqué -> FN (Background prédit)
                for tl in t_labels:
                    cm[int(tl), n_classes] += 1
                continue
                
            if len(t_boxes) == 0:
                # Tout ce qui est prédit est faux -> FP (Background réel)
                for pl in p_labels:
                    cm[n_classes, int(pl)] += 1
                continue
            
            # Calcul IoU Matrix
            ious = box_iou(t_boxes, p_boxes) # (n_targets, n_preds)
            
            # Matching Target -> Pred
            matched_p_indices = set()
            
            for t_idx in range(len(t_boxes)):
                # Trouver la meilleure pred pour ce target
                iou_row = ious[t_idx]
                max_iou, max_idx = iou_row.max(dim=0)
                max_idx = max_idx.item()
                
                if max_iou >= iou_thresh:
                    # Match trouvé !
                    if max_idx not in matched_p_indices:
                        true_cls = int(t_labels[t_idx])
                        pred_cls = int(p_labels[max_idx])
                        cm[true_cls, pred_cls] += 1
                        matched_p_indices.add(max_idx)
                    else:
                        # Déjà pris par un autre target (cas rare, double détection)
                        # On considère comme FN pour simplifier ici
                        cm[int(t_labels[t_idx]), n_classes] += 1
                else:
                    # Pas de match -> FN
                    cm[int(t_labels[t_idx]), n_classes] += 1
            
            # Les prédictions qui n'ont matché personne -> FP
            for p_idx in range(len(p_boxes)):
                if p_idx not in matched_p_indices:
                    cm[n_classes, int(p_labels[p_idx])] += 1

        # Plot Normalized
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        labels_plot = self.class_names + ['background']
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=labels_plot, yticklabels=labels_plot)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_folder / "confusion_matrix.png")
        plt.close()

    def _plot_label_distribution(self, targets, save_folder):
        all_labels = []
        for t in targets:
            all_labels.extend(t['labels'].tolist())
            
        plt.figure(figsize=(10, 6))
        # On mappe les IDs vers les Noms
        names = [self.class_names[int(l)] for l in all_labels if int(l) < len(self.class_names)]
        sns.countplot(x=names, palette="viridis")
        plt.title("Class Distribution in Validation Set")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_folder / "labels_distribution.jpg")
        plt.close()

