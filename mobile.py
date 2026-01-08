import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import tensorflow as tf

class YOLOv8TFLiteEvaluator:
    def __init__(self, model_path, img_size=640, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialise l'évaluateur YOLOv8 TFLite
        
        Args:
            model_path: Chemin vers le modèle .tflite
            img_size: Taille des images d'entrée
            conf_threshold: Seuil de confiance
            iou_threshold: Seuil IoU pour NMS
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Charger le modèle TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Stocker les prédictions et ground truths
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)
        
    def letterbox(self, img, new_shape=(640, 640)):
        """Redimensionne l'image avec padding pour garder le ratio"""
        shape = img.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw // 2, dh // 2
        
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, new_shape[0] - new_unpad[1] - dh
        left, right = dw, new_shape[1] - new_unpad[0] - dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, ratio, (dw, dh)
    
    def preprocess(self, img):
        """Prétraite l'image pour le modèle"""
        img, ratio, (dw, dh) = self.letterbox(img, (self.img_size, self.img_size))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0  # Normaliser 0-1
        if len(img.shape) == 3:
            img = img[None]  # Ajouter dimension batch
        return img, ratio, (dw, dh)
    
    def xywh2xyxy(self, x):
        """Convertit bbox de (x_center, y_center, w, h) vers (x1, y1, x2, y2)"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """Applique NMS sur les prédictions"""
        output = []
        for pred in prediction:
            pred = pred[pred[:, 4] > conf_thres]
            if not pred.shape[0]:
                output.append(np.zeros((0, 6)))
                continue
            
            boxes = self.xywh2xyxy(pred[:, :4])
            scores = pred[:, 4]
            classes = pred[:, 5:].argmax(1)
            
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                conf_thres,
                iou_thres
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                result = np.concatenate([
                    boxes[indices],
                    scores[indices:, None],
                    classes[indices:, None]
                ], axis=1)
                output.append(result)
            else:
                output.append(np.zeros((0, 6)))
        
        return output
    
    def predict(self, img_path):
        """Effectue une prédiction sur une image"""
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        
        img_input, ratio, (dw, dh) = self.preprocess(img)
        
        # Inférence
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Post-traitement
        output = output.transpose((0, 2, 1))
        predictions = self.non_max_suppression(output, self.conf_threshold, self.iou_threshold)
        
        # Rescale les coordonnées
        pred_boxes = []
        if len(predictions[0]) > 0:
            for det in predictions[0]:
                x1, y1, x2, y2, conf, cls = det
                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio
                pred_boxes.append([int(cls), conf, x1, y1, x2, y2])
        
        return pred_boxes
    
    def load_yolo_labels(self, label_path, img_w, img_h):
        """Charge les labels au format YOLO"""
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x_c, y_c, w, h = map(float, line.strip().split())
                    x1 = (x_c - w/2) * img_w
                    y1 = (y_c - h/2) * img_h
                    x2 = (x_c + w/2) * img_w
                    y2 = (y_c + h/2) * img_h
                    boxes.append([int(cls), x1, y1, x2, y2])
        return boxes
    
    def calculate_iou(self, box1, box2):
        """Calcule IoU entre deux boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate_dataset(self, images_dir, labels_dir):
        """Évalue le modèle sur un dataset"""
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"Évaluation sur {len(image_files)} images...")
        
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # Prédictions
            predictions = self.predict(img_path)
            
            # Ground truths
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            ground_truths = self.load_yolo_labels(label_path, img_w, img_h)
            
            # Stocker par classe
            for pred in predictions:
                cls = int(pred[0])
                self.predictions[cls].append(pred)
            
            for gt in ground_truths:
                cls = int(gt[0])
                self.ground_truths[cls].append(gt)
    
    def calculate_metrics(self, iou_threshold=0.5):
        """Calcule les métriques mAP, F1, P, R pour chaque classe"""
        classes = sorted(set(list(self.predictions.keys()) + list(self.ground_truths.keys())))
        
        results = {}
        for cls in classes:
            preds = sorted([p for p in self.predictions[cls]], key=lambda x: x[1], reverse=True)
            gts = self.ground_truths[cls]
            
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            matched_gts = set()
            
            for i, pred in enumerate(preds):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gts):
                    if j in matched_gts:
                        continue
                    iou = self.calculate_iou(pred[2:6], gt[1:5])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp[i] = 1
                    matched_gts.add(best_gt_idx)
                else:
                    fp[i] = 1
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / (len(gts) + 1e-16)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            
            # Calcul AP (Average Precision)
            recalls = np.concatenate(([0], recalls, [1]))
            precisions = np.concatenate(([0], precisions, [0]))
            
            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])
            
            indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
            
            # Métriques finales
            total_tp = np.sum(tp)
            total_fp = np.sum(fp)
            total_fn = len(gts) - total_tp
            
            precision = total_tp / (total_tp + total_fp + 1e-16)
            recall = total_tp / (total_tp + total_fn + 1e-16)
            f1 = 2 * precision * recall / (precision + recall + 1e-16)
            
            results[cls] = {
                'mAP@50': ap,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'TP': int(total_tp),
                'FP': int(total_fp),
                'FN': int(total_fn)
            }
        
        return results
    
    def save_results_to_excel(self, results, output_path='metrics_results.xlsx', class_names=None):
        """Sauvegarde les résultats dans un fichier Excel"""
        data = []
        for cls, metrics in results.items():
            class_name = class_names[cls] if class_names and cls < len(class_names) else f"Class_{cls}"
            data.append({
                'Classe': class_name,
                'mAP@50': metrics['mAP@50'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score'],
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN']
            })
        
        # Calcul des moyennes
        if data:
            avg_row = {
                'Classe': 'Moyenne',
                'mAP@50': np.mean([d['mAP@50'] for d in data]),
                'Precision': np.mean([d['Precision'] for d in data]),
                'Recall': np.mean([d['Recall'] for d in data]),
                'F1-Score': np.mean([d['F1-Score'] for d in data]),
                'TP': sum([d['TP'] for d in data]),
                'FP': sum([d['FP'] for d in data]),
                'FN': sum([d['FN'] for d in data])
            }
            data.append(avg_row)
        
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False, float_format='%.4f')
        print(f"\nRésultats sauvegardés dans: {output_path}")
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"c:\Users\BorisBob\Desktop\detection\models\exported\best_float32.tflite"  # Chemin vers votre modèle .tflite
    IMAGES_DIR = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\cotton_crop_dataset_ac_augmented\cotton_crop_yolo_augmented_dataset\test\images"  # Dossier des images de validation
    LABELS_DIR = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\cotton_crop_dataset_ac_augmented\cotton_crop_yolo_augmented_dataset\test\labels"  # Dossier des labels de validation
    OUTPUT_EXCEL = "evaluation_metrics_mobile.xlsx"
    
    # Noms des classes (à adapter selon votre dataset)
    CLASS_NAMES = ['A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp', 'Earias spp', 'Effet phyto', 'G. spodoctera', 'H. amirgera', 'Jasside', 'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron', 'S. derogata', 'S. frugiperda', 'Scarabees']

    # Créer l'évaluateur
    evaluator = YOLOv8TFLiteEvaluator(
        model_path=MODEL_PATH,
        img_size=640,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Évaluer le dataset
    evaluator.evaluate_dataset(IMAGES_DIR, LABELS_DIR)
    
    # Calculer les métriques
    results = evaluator.calculate_metrics(iou_threshold=0.5)
    
    # Sauvegarder dans Excel
    df = evaluator.save_results_to_excel(results, OUTPUT_EXCEL, CLASS_NAMES)
    
    # Afficher les résultats
    print("\n" + "="*80)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)