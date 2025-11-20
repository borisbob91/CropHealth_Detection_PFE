#!/usr/bin/env python3
"""
CropHealth Detection - YOLOv8n Inference
Prédictions + visualisations + métriques via Ultralytics (natif)
Conforme à Khan et al. (2025), Raza et al. (2025)

Usage:
    python yolo_predict.py --checkpoint runs/CropHealth_YOLOv8n_1107_1430/weights/best.pt \
                           --input data/test/images --output predictions/yolo \
                           --data-yaml data/yolo_crop/data.yaml
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO
from thop import profile
from torchinfo import summary


def compute_metrics_ultralytics(model, data_yaml):
    """
    Calcule métriques via Ultralytics framework (natif YOLOv8n)
    Conforme aux protocoles COCO intégrés dans Ultralytics
    """
    print(f"\n📊 Computing metrics via Ultralytics framework...")
    
    results = model.val(data=data_yaml, verbose=False)
    
    # Extraire métriques box (detection)
    box_metrics = results.box
    
    metrics = {
        'map50': box_metrics.map50 * 100,      # mAP@0.5
        'map': box_metrics.map * 100,          # mAP@0.5:0.95
        'map75': box_metrics.map75 * 100,      # mAP@0.75
        'precision': box_metrics.mp * 100,     # Mean Precision
        'recall': box_metrics.mr * 100,        # Mean Recall
    }
    
    # Calculer F1-Score
    p = metrics['precision'] / 100
    r = metrics['recall'] / 100
    metrics['f1'] = (2 * p * r / (p + r) * 100) if (p + r) > 0 else 0
    
    return metrics


def compute_fps_ultralytics(model, img_size=640, device='cuda', num_runs=100):
    """Mesure FPS moyen sur device"""
    dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_img, verbose=False)
    
    # Mesure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_img, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    return fps


def compute_model_size_gflops_yolo(model, img_size=640):
    """Calcule taille (MB) et GFLOPs via thop"""
    pt_model = model.model
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # GFLOPs
    macs, params = profile(pt_model, inputs=(dummy_input,), verbose=False)
    gflops = macs / 1e9
    
    # Taille
    model_stats = summary(pt_model, input_size=(1, 3, img_size, img_size), verbose=0)
    size_mb = model_stats.total_params * 4 / (1024 ** 2)
    
    return size_mb, gflops


def predict_and_save_ultralytics(model, input_dir, output_dir, conf_threshold=0.5):
    """Prédictions + visualisations via Ultralytics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Running predictions on {input_dir}...")
    
    # Prédictions (Ultralytics gère visualisation automatiquement)
    results = model.predict(
        source=input_dir,
        conf=conf_threshold,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
    )
    
    print(f"✅ Predictions saved to: {output_dir}")
    return results


def save_metrics_csv(metrics, output_path):
    """Sauvegarde métriques en CSV selon tableau rapport"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Métrique', 'Rôle', 'Valeur', 'Unité', 'Outil'])
        writer.writerow(['mAP@50', 'Principal', f"{metrics['map50']:.2f}", '%', 'Ultralytics'])
        writer.writerow(['mAP@50-95', 'Secondaire', f"{metrics['map']:.2f}", '%', 'Ultralytics'])
        writer.writerow(['mAP@75', 'Secondaire', f"{metrics['map75']:.2f}", '%', 'Ultralytics'])
        writer.writerow(['Precision', 'Analyse FP/FN', f"{metrics['precision']:.2f}", '%', 'Ultralytics'])
        writer.writerow(['Recall', 'Analyse FP/FN', f"{metrics['recall']:.2f}", '%', 'Ultralytics'])
        writer.writerow(['F1-Score', 'Analyse FP/FN', f"{metrics['f1']:.2f}", '%', 'Calculé (2PR/(P+R))'])
        writer.writerow(['FPS', 'Temps-réel', f"{metrics['fps']:.2f}", 'img/s', 'time.perf_counter()'])
        writer.writerow(['Taille', 'Edge', f"{metrics['size_mb']:.2f}", 'MB', 'torchinfo'])
        writer.writerow(['GFLOPs', 'Edge', f"{metrics['gflops']:.2f}", 'GFLOP', 'thop'])
    
    print(f"✅ Metrics saved to: {output_path}")


def main(args):
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - YOLOv8n Inference")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluator: Ultralytics (natif)")
    print(f"{'='*60}\n")
    
    # Charger modèle
    model = YOLO(args.checkpoint)
    
    # 1. Prédictions + visualisations
    if args.input:
        predict_and_save_ultralytics(model, args.input, args.output, args.conf)
    
    # 2. Métriques sur validation set
    if args.data_yaml:
        # Métriques Ultralytics
        metrics = compute_metrics_ultralytics(model, args.data_yaml)
        
        # FPS
        device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
        fps = compute_fps_ultralytics(model, img_size=640, device=device)
        metrics['fps'] = fps
        
        # Taille + GFLOPs
        size_mb, gflops = compute_model_size_gflops_yolo(model, img_size=640)
        metrics['size_mb'] = size_mb
        metrics['gflops'] = gflops
        total_params = sum(p.numel() for p in model.model.parameters())
        metrics['total_params'] = total_params
        # Afficher
        print(f"\n📈 Results (Ultralytics):")
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
        csv_path = Path(args.output) / 'yolo_metrics.csv'
        save_metrics_csv(metrics, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropHealth YOLOv8n Inference (Ultralytics)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt (YOLOv8n weights)')
    parser.add_argument('--input', type=str,
                        help='Input images directory for predictions')
    parser.add_argument('--output', type=str, default='predictions/yolo',
                        help='Output directory for visualizations')
    parser.add_argument('--data-yaml', type=str,
                        help='Path to data.yaml (for Ultralytics validation)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)