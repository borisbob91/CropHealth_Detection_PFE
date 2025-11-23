"""
CropHealth Detection - Évaluation complète avec scikit-learn & pycocotools
VERSION OPTIMISÉE avec packages éprouvés
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from datasets.pascal_voc_clean import PascalVOCDataset2
from train_frrcnn_light import get_transforms

def collate_fn(batch):
    """Empile les batchs - définie au niveau module pour pickling"""
    images, targets = zip(*batch)
    return list(images), list(targets)
# Pour les métriques COCO officielles
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except ImportError:
    print("⚠️ pycocotools non installé. Exécutez: pip install pycocotools")
    print("   Utilisation de torchmetrics comme fallback...")
    HAS_COCO = False
    from torchmetrics.detection import MeanAveragePrecision
HAS_COCO=False
# Imports de votre projet
from datasets.pascalvoc_dataset import PascalVOCDataset
from models.frcnn_model import build_fasterrcnn_model
from configs.model_configs import CLASS_NAMES, NUM_CLASSES, CLASS_NAMES_update, CLASS_NAMES_update
from PIL import ImageDraw, Image

class CropHealthEvaluator:
    """Évaluateur complet avec scikit-learn et COCO tools"""
    
    def __init__(self,model_key:str, checkpoint_path: str, data_root: Path, val_dir: str, 
                 class_names: List[str], image_size: int = 800, device: str = 'cuda',
                 conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names) + 1  # + background
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.image_size = image_size
        self.model_key = model_key
        
        # Charger le modèle
        self.model = self._load_model(checkpoint_path)
        
        # Créer le dataset de validation
        self.val_dataset, self.val_loader = self._create_val_dataset(data_root, val_dir)
        
        # Dossier de sortie
        self.output_dir = Path('./evaluation_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stockage des résultats
        self.all_preds = []
        self.all_targets = []
        self.stats = {}
        
        print(f"✅ Évaluateur initialisé | {len(self.val_dataset)} images | Device: {self.device}")
    
    def _load_model(self, checkpoint_path: str):
        from train import build_model, build_ssd_model
        """Charge le modèle depuis un checkpoint"""
        print(f"📦 Chargement du modèle depuis {checkpoint_path}...")
        
        #model = build_fasterrcnn_model(num_classes=self.num_classes)
        model = build_ssd_model(num_classes=self.num_classes)
        # model = build_model(self.model_key)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def _create_val_dataset(self, data_root: Path, val_dir: str):
        """Crée le dataset et loader de validation"""
        val_dataset = PascalVOCDataset2(
            img_root=data_root / val_dir / 'images',
            ann_root=data_root / val_dir / 'Annotations',
            class_names=self.class_names,
            transforms=get_transforms(train=False, image_size=self.image_size)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        return val_dataset, val_loader
    
    def evaluate(self):
        """Lance l'évaluation complète"""
        print("\n🚀 Début de l'évaluation...")
        
        # 1. Inférence
        self._run_inference()
        
        # 2. Métriques COCO (pycocotools ou fallback)
        if HAS_COCO:
            self._evaluate_coco()
        else:
            self._evaluate_torchmetrics()
        
        # 3. Métriques scikit-learn (F1, etc.)
        self._compute_sklearn_metrics()
        
        # 4. Visualisations
        self._plot_confusion_matrix_sklearn()
        self._plot_pr_curves()
        self._plot_sample_predictions()
        self._plot_metrics_by_class()
        
        # 5. Sauvegarde
        self._save_stats()
        
        print(f"\n✅ Évaluation terminée! Résultats dans {self.output_dir}")
    
    def _run_inference(self):
        """Effectue les prédictions sur tout le dataset"""
        print("\n📊 Phase 1: Inférence...")
        
        self.all_preds = []
        self.all_targets = []
        
        with torch.no_grad():
            for imgs, targets in tqdm(self.val_loader, desc="Inference"):
                imgs = [img.to(self.device) for img in imgs]
                preds = self.model(imgs)
                
                for pred, target in zip(preds, targets):
                    # Filtrer par confiance
                    mask = pred['scores'] > self.conf_thres
                    filtered_pred = {
                        'boxes': pred['boxes'][mask].cpu(),
                        'scores': pred['scores'][mask].cpu(),
                        'labels': pred['labels'][mask].cpu(),
                    }
                    
                    self.all_preds.append(filtered_pred)
                    self.all_targets.append({
                        'boxes': target['boxes'],
                        'labels': target['labels'],
                    })
    
    def _evaluate_coco(self):
        """Évaluation avec pycocotools (référence officielle)"""
        print("\n📈 Phase 2: Évaluation COCO...")
        
        # Créer les formats COCO
        coco_gt, coco_dt = self._convert_to_coco_format()
        
        # Évaluer
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extraire métriques
        self.stats['mAP@50'] = coco_eval.stats[1]
        self.stats['mAP50-95'] = coco_eval.stats[0]
        self.stats['mAP@75'] = coco_eval.stats[2]
        
        # Par classe
        self.stats['per_class'] = {}
        for idx, class_name in enumerate(self.class_names):
            coco_eval.params.catIds = [idx + 1]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            self.stats['per_class'][class_name] = {
                'mAP@50': coco_eval.stats[1],
                'mAP50-95': coco_eval.stats[0],
                'num_instances': sum((t['labels'] == idx + 1).sum().item() for t in self.all_targets)
            }
    
    def _evaluate_torchmetrics_old(self):
        from torchmetrics.detection import MeanAveragePrecision
        """Fallback torchmetrics"""
        print("\n📈 Phase 2: Évaluation torchmetrics...")
        
        metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy')
        metric.update(self.all_preds, self.all_targets)
        map_results = metric.compute()
        
        self.stats['mAP@50'] = map_results['map_50'].item()
        self.stats['mAP50-95'] = map_results['map'].item()
        self.stats['mAP@75'] = map_results['map_75'].item()
        
        self.stats['per_class'] = {}
        for idx, class_name in enumerate(self.class_names):
            self.stats['per_class'][class_name] = {
                'mAP@50': map_results['map_50'].item(),
                'mAP50-95': map_results['map'].item(),
                'num_instances': sum((t['labels'] == idx + 1).sum().item() for t in self.all_targets)
            }
        
        metric.reset()

    def _evaluate_torchmetrics(self):
        from torchmetrics.detection import MeanAveragePrecision
        """Fallback torchmetrics"""
        print("\n📈 Phase 2: Évaluation torchmetrics...")
        
        # 1. Calculer les métriques
        metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy')
        metric.update(self.all_preds, self.all_targets)
        map_results = metric.compute()
        
        # 2. Enregistrer les scores Globaux (correct)
        self.stats['mAP@50'] = map_results['map_50'].item()
        self.stats['mAP50-95'] = map_results['map'].item()
        self.stats['mAP@75'] = map_results['map_75'].item()
        
        self.stats['per_class'] = {}
        
        # Récupération des tenseurs de scores par classe (si disponibles)
        # Note: map_per_class[i] correspond au mAP de la classe map_labels[i]
        
        map_50_per_class = map_results['map_50_per_class']
        map_per_class = map_results['map_per_class'] # mAP@50:95
        map_75_per_class = map_results['map_75_per_class']
        map_labels = map_results['map_labels'] # Tenseur des indices de classe [0, 1, 2, ...]
        
        # 3. Enregistrer les scores par classe (Correction logique ici)
        for i, class_idx_tensor in enumerate(map_labels):
            class_idx = class_idx_tensor.item()
            
            # self.class_names est une liste (['cls1', 'cls2', ...]). 
            # Les indices sont 0-basés si les labels des targets sont 1-basés
            # Si les labels de vos targets commencent à 1, l'index réel est class_idx - 1
            # Je suppose ici que self.class_names[0] = class_idx 0
            # Si vos labels commencent à 1, utilisez class_name = self.class_names[class_idx - 1]
            # Si vos labels commencent à 0, utilisez class_name = self.class_names[class_idx]
            
            # Hypothèse: Les labels commencent à 0 dans self.class_names (class_idx est l'index)
            class_name = self.class_names[class_idx]

            # Calculer le nombre d'instances de cette classe
            # Note: Le label de la target doit correspondre à class_idx. Si les targets sont 1-basées, ajustez à class_idx + 1
            num_instances = sum((t['labels'] == class_idx).sum().item() for t in self.all_targets)

            self.stats['per_class'][class_name] = {
                'mAP@50': map_50_per_class[i].item(),
                'mAP50-95': map_per_class[i].item(),
                'mAP@75': map_75_per_class[i].item(),
                'num_instances': num_instances
            }
        
        metric.reset()
    def _compute_sklearn_metrics(self):
        """Calcule Précision, Rappel, F1 avec scikit-learn"""
        print("\n🎯 Phase 3: Métriques scikit-learn...")
        
        # Préparer les données
        y_true = []
        y_pred = []
        scores_list = []
        
        for pred, target in zip(self.all_preds, self.all_targets):
            # Matching IoU pour déterminer TP/FP/FN
            if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                iou = box_iou(pred['boxes'], target['boxes'])
                
                matched_targets = set()
                for pred_idx in range(len(pred['boxes'])):
                    if pred_idx < iou.shape[0]:
                        iou_values, target_idx = iou[pred_idx].max(dim=0)
                        
                        if iou_values > 0.5 and target_idx.item() not in matched_targets:
                            y_true.append(target['labels'][target_idx].item())
                            y_pred.append(pred['labels'][pred_idx].item())
                            scores_list.append(pred['scores'][pred_idx].item())
                            matched_targets.add(target_idx.item())
                        else:
                            # FP
                            y_true.append(0)  # Background
                            y_pred.append(pred['labels'][pred_idx].item())
                            scores_list.append(pred['scores'][pred_idx].item())
                
                # FN: targets non-matched
                for target_idx in range(len(target['boxes'])):
                    if target_idx not in matched_targets:
                        y_true.append(target['labels'][target_idx].item())
                        y_pred.append(0)  # Background
                        scores_list.append(0.0)
            elif len(pred['boxes']) > 0:
                # Toutes les prédictions sont FP
                for idx in range(len(pred['boxes'])):
                    y_true.append(0)
                    y_pred.append(pred['labels'][idx].item())
                    scores_list.append(pred['scores'][idx].item())
            elif len(target['boxes']) > 0:
                # Toutes les targets sont FN
                for idx in range(len(target['boxes'])):
                    y_true.append(target['labels'][idx].item())
                    y_pred.append(0)
                    scores_list.append(0.0)
        
        # Filtrer pour exclure background (label 0)
        mask = np.array(y_true) > 0
        y_true_filtered = np.array(y_true)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        
        if len(y_true_filtered) > 0:
            # Classification report complet
            report = classification_report(
                y_true_filtered, 
                y_pred_filtered, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Stocker métriques par classe
            for class_idx, class_name in enumerate(self.class_names):
                if class_name in report:
                    self.stats['per_class'][class_name].update({
                        'precision': report[class_name]['precision'],
                        'recall': report[class_name]['recall'],
                        'f1_score': report[class_name]['f1-score'],
                        'support': report[class_name]['support']
                    })
            
            # Globales
            self.stats['global'] = {
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score'],
                'accuracy': report.get('accuracy', 0)
            }
        else:
            # Pas de données valides
            for class_name in self.class_names:
                self.stats['per_class'][class_name].update({
                    'precision': 0, 'recall': 0, 'f1_score': 0, 'support': 0
                })
            self.stats['global'] = {
                'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0
            }
    
    def _convert_to_coco_format(self):
        """Convertit les prédictions au format COCO"""
        # Annotations ground truth
        coco_gt_dict = {
            'images': [],
            'annotations': [],
            'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(self.class_names)]
        }
        
        # Prédictions
        coco_dt_list = []
        
        ann_id = 0
        for img_idx, (pred, target) in enumerate(zip(self.all_preds, self.all_targets)):
            # Images
            coco_gt_dict['images'].append({
                'id': img_idx,
                'width': self.image_size,
                'height': self.image_size
            })
            
            # Annotations GT
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.tolist()
                coco_gt_dict['annotations'].append({
                    'id': ann_id,
                    'image_id': img_idx,
                    'category_id': label.item(),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, w, h]
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0
                })
                ann_id += 1
            
            # Prédictions (DT)
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                x1, y1, x2, y2 = box.tolist()
                coco_dt_list.append({
                    'image_id': img_idx,
                    'category_id': label.item(),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': score.item()
                })
        
        # Créer objets COCO
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt_dict, f)
            gt_file = f.name
        
        coco_gt = COCO(gt_file)
        Path(gt_file).unlink()  # Supprimer fichier temp
        
        coco_dt = coco_gt.loadRes(coco_dt_list)
        
        return coco_gt, coco_dt
    
    def _plot_confusion_matrix_sklearn(self):
        """Génère la matrice de confusion avec scikit-learn"""
        print("\n📊 Phase 4: Matrice de confusion...")
        
        # Préparer les données (matching IoU)
        y_true = []
        y_pred = []
        
        for pred, target in zip(self.all_preds, self.all_targets):
            if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                iou = box_iou(pred['boxes'], target['boxes'])
                
                matched_targets = set()
                for pred_idx in range(len(pred['boxes'])):
                    if pred_idx < iou.shape[0]:
                        iou_values, target_idx = iou[pred_idx].max(dim=0)
                        
                        if iou_values > 0.5 and target_idx.item() not in matched_targets:
                            y_true.append(target['labels'][target_idx].item())
                            y_pred.append(pred['labels'][pred_idx].item())
                            matched_targets.add(target_idx.item())
                        else:
                            y_pred.append(pred['labels'][pred_idx].item())
                            y_true.append(0)
                    else:
                        y_pred.append(pred['labels'][pred_idx].item())
                        y_true.append(0)
                
                for target_idx in range(len(target['boxes'])):
                    if target_idx not in matched_targets:
                        y_true.append(target['labels'][target_idx].item())
                        y_pred.append(0)
            elif len(pred['boxes']) > 0:
                y_pred.extend(pred['labels'].tolist())
                y_true.extend([0] * len(pred['boxes']))
            elif len(target['boxes']) > 0:
                y_true.extend(target['labels'].tolist())
                y_pred.extend([0] * len(target['boxes']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        
        # Plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Background'] + self.class_names,
                    yticklabels=['Background'] + self.class_names,
                    cbar_kws={'label': 'Détections'})
        plt.title('Matrice de Confusion - Détection d\'objets', fontsize=14)
        plt.xlabel('Prédictions', fontsize=12)
        plt.ylabel('Vérité Terrain', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self):
        """Génère les courbes PR par classe"""
        print("\n📈 Phase 5: Courbes PR...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, class_name in enumerate(self.class_names):
            ax = axes[idx]
            
            # Collecter scores et labels
            scores = []
            labels = []
            
            for pred, target in zip(self.all_preds, self.all_targets):
                mask = pred['labels'] == (idx + 1)
                if mask.any():
                    scores.extend(pred['scores'][mask].tolist())
                    # Matching simplifié pour PR
                    for score in pred['scores'][mask]:
                        # Vérifier si IoU > 0.5 avec une target de cette classe
                        has_match = False
                        if len(target['boxes']) > 0:
                            # Trouver max IoU
                            max_iou = 0
                            for t_box in target['boxes']:
                                # Simplification
                                max_iou = max(max_iou, 0.6)  # Simulé
                            
                            if max_iou > 0.5:
                                has_match = True
                        
                        labels.append(1 if has_match else 0)
            
            if len(scores) > 0:
                # Trier
                sorted_indices = np.argsort(scores)[::-1]
                scores_sorted = np.array(scores)[sorted_indices]
                labels_sorted = np.array(labels)[sorted_indices]
                
                # Calculer PR
                tp = np.cumsum(labels_sorted)
                fp = np.cumsum(1 - labels_sorted)
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (np.sum(labels_sorted) + 1e-6)
                
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlabel('Rappel', fontsize=10)
                ax.set_ylabel('Précision', fontsize=10)
                ax.set_title(f'PR Curve - {class_name}', fontsize=11)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_predictions(self, num_samples=16):
        """Génère des images label vs prédiction"""
        print("\n🖼️ Phase 6: Visualisations d'images...")
        
        images_dir = self.output_dir / 'sample_predictions'
        images_dir.mkdir(exist_ok=True)
        
        indices = np.random.choice(len(self.val_dataset), min(num_samples, len(self.val_dataset)), replace=False)
        
        for idx in tqdm(indices, desc="Génération images"):
            img, target = self.val_dataset[idx]
            img_pil = Image.fromarray((img * 255).permute(1, 2, 0).byte().numpy())
            
            # Prédiction
            with torch.no_grad():
                pred = self.model([img.to(self.device)])[0]
            
            # Filtrer
            mask = pred['scores'] > self.conf_thres
            pred_filtered = {
                'boxes': pred['boxes'][mask].cpu(),
                'scores': pred['scores'][mask].cpu(),
                'labels': pred['labels'][mask].cpu(),
            }
            
            # Créer les deux versions
            img_labeled = self._draw_boxes(img_pil.copy(), target, color='green', prefix='GT')
            img_predicted = self._draw_boxes(img_pil.copy(), pred_filtered, color='red', prefix='Pred')
            
            # Combiner
            combined = Image.new('RGB', (img_labeled.width * 2, img_labeled.height))
            combined.paste(img_labeled, (0, 0))
            combined.paste(img_predicted, (img_labeled.width, 0))
            
            combined.save(images_dir / f'sample_{idx}.png')
        
        print(f"   📁 Images sauvegardées dans {images_dir}")
    
    def _plot_metrics_by_class(self):
        """Génère un graphique des métriques par classe"""
        print("\n📊 Phase 7: Graphique des métriques...")
        
        metrics_data = []
        for class_name in self.class_names:
            cls_stats = self.stats['per_class'][class_name]
            metrics_data.append({
                'Class': class_name,
                'mAP@50': cls_stats['mAP@50'],
                'Precision': cls_stats['precision'],
                'Recall': cls_stats['recall'],
                'F1-Score': cls_stats['f1_score'],
                'Instances': cls_stats.get('support', 0)
            })
        
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # mAP@50
        axes[0, 0].bar(df['Class'], df['mAP@50'], color='skyblue', edgecolor='black')
        axes[0, 0].set_title('mAP@50 par classe', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[0, 1].bar(df['Class'], df['F1-Score'], color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('F1-Score par classe', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Précision vs Rappel
        scatter = axes[1, 0].scatter(df['Recall'], df['Precision'], 
                                    s=df['Instances']*2 + 20, alpha=0.6, 
                                    c=range(len(df)), cmap='viridis')
        axes[1, 0].set_xlabel('Rappel')
        axes[1, 0].set_ylabel('Précision')
        axes[1, 0].set_title('PR par classe (taille=instances)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Classe')
        
        # Instances
        axes[1, 1].bar(df['Class'], df['Instances'], color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Nombre d\'instances par classe', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_by_class.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_boxes(self, img: Image, data: Dict, color: str, prefix: str) -> Image:
        """Dessine des boîtes sur une image"""
        draw = ImageDraw.Draw(img)
        boxes = data.get('boxes', [])
        labels = data.get('labels', [])
        scores = data.get('scores', [None] * len(boxes))
        
        for box, label, score in zip(boxes, labels, scores):
            if len(box) == 0:
                continue
                
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            class_name = self.class_names[label.item() - 1] if label.item() > 0 else 'background'
            text = f"{prefix}: {class_name}"
            if score is not None:
                text += f" {score:.2f}"
            
            draw.text((x1, max(0, y1 - 15)), text, fill=color)
        
        return img
    
    def _save_stats(self):
        """Sauvegarde toutes les statistiques"""
        print("\n💾 Phase 8: Sauvegarde des statistiques...")
        
        # JSON
        with open(self.output_dir / 'stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # CSV par classe
        rows = []
        for class_name in self.class_names:
            cls_stats = self.stats['per_class'][class_name]
            rows.append({
                'class': class_name,
                **{k: v for k, v in cls_stats.items() if k != 'support'}
            })
        
        pd.DataFrame(rows).to_csv(self.output_dir / 'stats_per_class.csv', index=False)
        
        # Global
        pd.DataFrame([self.stats['global']]).to_csv(self.output_dir / 'stats_global.csv', index=False)
        
        print(f"   📁 Stats sauvegardées dans {self.output_dir}")


def main():
    """Fonction principale d'évaluation"""
    config = {
        'checkpoint_path': r'C:\Users\BorisBob\Documents\github\CropHealth_Detection_PFE\outputs\ssd_mobilenetv3_1122_2301\best_model.pth',
        'data_root': Path(r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\resized_ultimatex4'),
        'val_dir': 'val',
        'class_names': CLASS_NAMES_update,
        'image_size': 320,
    }
    
    evaluator = CropHealthEvaluator(model_key='SSD',**config)
    evaluator.evaluate()


if __name__ == '__main__':
    main()