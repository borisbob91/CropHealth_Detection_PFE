# 🌾 CropHealth Detection - Workflow Complet Google Colab

**Notebook pour exécuter l'ensemble du projet CropHealth Detection sur Google Colab**

Copiez-collez les cellules suivantes dans un nouveau notebook Colab.

---

## 📌 **Cellule 1 : Configuration GPU**

```python
# Vérifier GPU disponible
import torch

print("="*60)
print("🔧 GPU Configuration")
print("="*60)

if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected - using CPU")
    print("💡 Go to Runtime > Change runtime type > GPU")

print("="*60)
```

---

## 📌 **Cellule 2 : Cloner le repository**

```python
# Cloner le projet depuis GitHub
!git clone https://github.com/borisbob91/CropHealth_Detection_PFE.git
%cd CropHealth_Detection_PFE

# Vérifier structure
!ls -la
```

---

## 📌 **Cellule 3 : Installer les dépendances**

```python
# Installation des dépendances
print("📦 Installing dependencies...")

!pip install -q torch torchvision torchmetrics
!pip install -q albumentations ultralytics
!pip install -q effdet timm tensorboard
!pip install -q pycocotools scikit-learn
!pip install -q thop torchinfo matplotlib seaborn

print("✅ Dependencies installed!")

# Vérifier installations
import torch
import torchvision
import ultralytics
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ TorchVision: {torchvision.__version__}")
print(f"✅ Ultralytics: {ultralytics.__version__}")
```

---

## 📌 **Cellule 4 : Monter Google Drive (optionnel)**

```python
# Monter Google Drive pour accéder au dataset
from google.colab import drive
drive.mount('/content/drive')

# Créer lien symbolique vers dataset dans Drive
# Adapter le chemin selon votre structure Drive
!ln -s /content/drive/MyDrive/CropHealth_Data /content/CropHealth_Detection_PFE/data

print("✅ Google Drive mounted!")
print("📁 Dataset path: /content/CropHealth_Detection_PFE/data")
```

---

## 📌 **Cellule 5 : Uploader dataset (alternative)**

```python
# Si dataset pas dans Drive, uploader ZIP
from google.colab import files
import zipfile

print("📤 Upload your dataset ZIP file...")
uploaded = files.upload()

# Décompresser
for filename in uploaded.keys():
    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data/')
    
print("✅ Dataset extracted to data/")

# Vérifier structure
!tree data/ -L 2 -d
```

---

## 📌 **Cellule 6 : Vérifier structure dataset**

```python
# Vérifier que le dataset est bien structuré
import os
from pathlib import Path

data_root = Path('data/yolo_crop')

print("="*60)
print("📁 Dataset Structure Check")
print("="*60)

required_paths = [
    data_root / 'train' / 'images',
    data_root / 'train' / 'labels',
    data_root / 'val' / 'images',
    data_root / 'val' / 'labels',
]

for path in required_paths:
    if path.exists():
        count = len(list(path.glob('*')))
        print(f"✅ {path.relative_to(data_root)}: {count} files")
    else:
        print(f"❌ {path.relative_to(data_root)}: NOT FOUND")

print("="*60)
```

---

## 📌 **Cellule 7 : Training SSD MobileNetV3**

```python
# Train SSD
!python train.py \
    --model ssd \
    --data data/yolo_crop \
    --device cuda

print("\n✅ SSD training complete!")
print("📁 Results in: runs/CropHealth_SSD_*/")
```

---

## 📌 **Cellule 8 : Training YOLOv8n**

```python
# Train YOLOv8n
!python train_yolo.py \
    --data data/yolo_crop/data.yaml \
    --device 0 \
    --cache

print("\n✅ YOLOv8n training complete!")
print("📁 Results in: runs/CropHealth_YOLOv8n_*/")
```

---

## 📌 **Cellule 9 : Training EfficientDet-D0**

```python
# Convertir YOLO → COCO
!python utils/yolo2coco.py \
    --yolo-root data/yolo_crop \
    --output data/coco_crop

# Train EfficientDet
!python train.py \
    --model efficientdet \
    --data data/coco_crop \
    --device cuda

print("\n✅ EfficientDet-D0 training complete!")
print("📁 Results in: runs/CropHealth_EfficientDet_*/")
```

---

## 📌 **Cellule 10 : Training Faster R-CNN**

```python
# Train Faster R-CNN ResNet50
!python train.py \
    --model fasterrcnn \
    --data data/yolo_crop \
    --device cuda

print("\n✅ Faster R-CNN training complete!")
print("📁 Results in: runs/CropHealth_FasterRCNN_*/")
```

---

## 📌 **Cellule 11 : Training Faster R-CNN Light**

```python
# Train Faster R-CNN MobileNetV3 (light)
!python train.py \
    --model fasterrcnn_light \
    --data data/yolo_crop \
    --device cuda

print("\n✅ Faster R-CNN Light training complete!")
print("📁 Results in: runs/CropHealth_FasterRCNN_light_*/")
```

---

## 📌 **Cellule 12 : TensorBoard (pendant training)**

```python
# Lancer TensorBoard pour visualiser métriques
%load_ext tensorboard
%tensorboard --logdir runs/

# Ou spécifier un run particulier
# %tensorboard --logdir runs/CropHealth_SSD_1117_1430/
```

---

## 📌 **Cellule 13 : Inférence SSD**

```python
# Trouver le dernier checkpoint SSD
import glob
ssd_checkpoints = sorted(glob.glob('runs/CropHealth_SSD_*/best.pt'))
latest_ssd = ssd_checkpoints[-1] if ssd_checkpoints else None

if latest_ssd:
    print(f"📦 Using checkpoint: {latest_ssd}")
    
    # Inférence avec métriques
    !python predict.py \
        --model ssd \
        --checkpoint {latest_ssd} \
        --input data/yolo_crop/val/images \
        --val-data data/yolo_crop \
        --output predictions/ssd \
        --conf 0.5
    
    print("\n✅ Predictions saved to predictions/ssd/")
else:
    print("❌ No SSD checkpoint found. Train the model first.")
```

---

## 📌 **Cellule 14 : Inférence YOLOv8n**

```python
# Trouver le dernier checkpoint YOLOv8n
yolo_checkpoints = sorted(glob.glob('runs/CropHealth_YOLOv8n_*/weights/best.pt'))
latest_yolo = yolo_checkpoints[-1] if yolo_checkpoints else None

if latest_yolo:
    print(f"📦 Using checkpoint: {latest_yolo}")
    
    # Inférence
    !python yolo_predict.py \
        --checkpoint {latest_yolo} \
        --input data/yolo_crop/val/images \
        --data-yaml data/yolo_crop/data.yaml \
        --output predictions/yolo \
        --conf 0.5
    
    print("\n✅ Predictions saved to predictions/yolo/")
else:
    print("❌ No YOLOv8n checkpoint found. Train the model first.")
```

---

## 📌 **Cellule 15 : Évaluation multi-modèles**

```python
# Évaluation comparative de tous les modèles
!python evaluate_models.py \
    --checkpoints \
        ssd:runs/CropHealth_SSD_*/best.pt \
        yolov8n:runs/CropHealth_YOLOv8n_*/weights/best.pt \
        efficientdet:runs/CropHealth_EfficientDet_*/best.pt \
        fasterrcnn:runs/CropHealth_FasterRCNN_*/best.pt \
        fasterrcnn_light:runs/CropHealth_FasterRCNN_light_*/best.pt \
    --val-data data/yolo_crop \
    --output evaluation_results \
    --device cuda

print("\n✅ Evaluation complete!")
print("📊 Results in: evaluation_results/")
```

---

## 📌 **Cellule 16 : Visualiser résultats évaluation**

```python
# Afficher les plots générés
from IPython.display import Image, display
import glob

print("="*60)
print("📊 EVALUATION RESULTS")
print("="*60)

# mAP@50 comparison
if Path('evaluation_results/map50_comparison.png').exists():
    print("\n📈 mAP@50 Global Comparison:")
    display(Image('evaluation_results/map50_comparison.png'))

# AP@50 per class
if Path('evaluation_results/ap50_per_class_comparison.png').exists():
    print("\n📈 AP@50 per Class:")
    display(Image('evaluation_results/ap50_per_class_comparison.png'))

# F1-Score per class
if Path('evaluation_results/f1_per_class_comparison.png').exists():
    print("\n📈 F1-Score per Class:")
    display(Image('evaluation_results/f1_per_class_comparison.png'))

# Confusion matrices
cm_files = sorted(glob.glob('evaluation_results/confusion_matrix_*.png'))
for cm_file in cm_files[:3]:  # Afficher 3 premiers
    model_name = Path(cm_file).stem.replace('confusion_matrix_', '')
    print(f"\n📊 Confusion Matrix - {model_name}:")
    display(Image(cm_file))
```

---

## 📌 **Cellule 17 : Afficher CSV métriques**

```python
# Afficher tableaux CSV
import pandas as pd

print("="*60)
print("📊 METRICS SUMMARY")
print("="*60)

# Global metrics
if Path('evaluation_results/global_metrics.csv').exists():
    print("\n📈 Global Metrics (mAP@50 + F1-Score):")
    df_global = pd.read_csv('evaluation_results/global_metrics.csv')
    display(df_global)

# AP@50 per class
if Path('evaluation_results/ap50_per_class.csv').exists():
    print("\n📈 AP@50 per Class:")
    df_ap = pd.read_csv('evaluation_results/ap50_per_class.csv')
    display(df_ap)

# F1 per class
if Path('evaluation_results/f1_per_class.csv').exists():
    print("\n📈 F1-Score per Class:")
    df_f1 = pd.read_csv('evaluation_results/f1_per_class.csv')
    display(df_f1.head(20))  # Premières lignes
```

---

## 📌 **Cellule 18 : Export ONNX**

```python
# Export best model to ONNX
best_checkpoint = 'runs/CropHealth_SSD_*/best.pt'  # Adapter selon meilleur modèle

!python export/export_models.py \
    --model ssd \
    --checkpoint {best_checkpoint} \
    --format onnx \
    --output exports/ssd

print("\n✅ ONNX export complete!")
print("📦 Model: exports/ssd/CropHealth_SSD.onnx")
```

---

## 📌 **Cellule 19 : Export TFLite INT8**

```python
# Export to TFLite INT8 for mobile deployment
!python export/export_models.py \
    --model ssd \
    --checkpoint {best_checkpoint} \
    --format tflite \
    --quantize int8 \
    --calibration-data data/yolo_crop/train/images \
    --output exports/ssd

print("\n✅ TFLite INT8 export complete!")
print("📦 Model: exports/ssd/CropHealth_SSD_int8.tflite")
```

---

## 📌 **Cellule 20 : Benchmark exports**

```python
# Benchmark PyTorch vs ONNX
!python export/benchmark_exports.py \
    --model ssd \
    --pytorch {best_checkpoint} \
    --onnx exports/ssd/CropHealth_SSD.onnx \
    --tflite exports/ssd/CropHealth_SSD_int8.tflite \
    --runs 100 \
    --output benchmark_ssd.csv

# Afficher résultats
print("\n📊 Benchmark Results:")
df_bench = pd.read_csv('benchmark_ssd.csv')
display(df_bench)
```

---

## 📌 **Cellule 21 : Visualiser prédictions**

```python
# Afficher quelques prédictions
from IPython.display import Image as IPImage, display
import random

pred_images = list(Path('predictions/ssd/').glob('*.jpg'))
random.shuffle(pred_images)

print("="*60)
print("🖼️  SAMPLE PREDICTIONS")
print("="*60)

for img_path in pred_images[:5]:  # 5 images aléatoires
    print(f"\n📷 {img_path.name}:")
    display(IPImage(filename=str(img_path), width=600))
```

---

## 📌 **Cellule 22 : Télécharger résultats**

```python
# Zipper tous les résultats pour téléchargement
import shutil

# Créer archive
print("📦 Creating results archive...")

shutil.make_archive('crophealth_results', 'zip', 'runs')
shutil.make_archive('crophealth_predictions', 'zip', 'predictions')
shutil.make_archive('crophealth_evaluation', 'zip', 'evaluation_results')
shutil.make_archive('crophealth_exports', 'zip', 'exports')

print("✅ Archives created!")

# Télécharger
from google.colab import files

print("\n📥 Downloading archives...")
files.download('crophealth_results.zip')
files.download('crophealth_predictions.zip')
files.download('crophealth_evaluation.zip')
files.download('crophealth_exports.zip')

print("\n✅ Download complete!")
```

---

## 📌 **Cellule 23 : Sauvegarder dans Google Drive**

```python
# Copier résultats vers Google Drive
import shutil
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
drive_backup = f'/content/drive/MyDrive/CropHealth_Backup_{timestamp}'

print(f"💾 Backing up to: {drive_backup}")

# Créer dossier
!mkdir -p {drive_backup}

# Copier
shutil.copytree('runs', f'{drive_backup}/runs', dirs_exist_ok=True)
shutil.copytree('predictions', f'{drive_backup}/predictions', dirs_exist_ok=True)
shutil.copytree('evaluation_results', f'{drive_backup}/evaluation_results', dirs_exist_ok=True)
shutil.copytree('exports', f'{drive_backup}/exports', dirs_exist_ok=True)

print(f"\n✅ Backup complete!")
print(f"📁 Location: {drive_backup}")
```

---

## 📌 **Cellule 24 : Résumé final**

```python
# Afficher résumé complet
import json

print("="*60)
print("🌾 CROPHEALTH DETECTION - FINAL SUMMARY")
print("="*60)

# Compter checkpoints
models_trained = {
    'SSD': len(glob.glob('runs/CropHealth_SSD_*/best.pt')),
    'YOLOv8n': len(glob.glob('runs/CropHealth_YOLOv8n_*/weights/best.pt')),
    'EfficientDet': len(glob.glob('runs/CropHealth_EfficientDet_*/best.pt')),
    'Faster R-CNN': len(glob.glob('runs/CropHealth_FasterRCNN_*/best.pt')),
    'Faster R-CNN Light': len(glob.glob('runs/CropHealth_FasterRCNN_light_*/best.pt')),
}

print("\n📊 Models Trained:")
for model, count in models_trained.items():
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {model}: {count} checkpoint(s)")

# Exports
print("\n📦 Models Exported:")
export_formats = ['onnx', 'tflite', 'engine']
for fmt in export_formats:
    count = len(glob.glob(f'exports/**/*.{fmt}', recursive=True))
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {fmt.upper()}: {count} file(s)")

# Évaluations
print("\n📈 Evaluation Results:")
eval_files = [
    'global_metrics.csv',
    'ap50_per_class.csv',
    'f1_per_class.csv',
    'map50_comparison.png'
]
for file in eval_files:
    path = Path('evaluation_results') / file
    status = "✅" if path.exists() else "❌"
    print(f"  {status} {file}")

print("\n" + "="*60)
print("✅ Workflow Complete!")
print("="*60)
```

---

## 🎯 **Instructions d'utilisation**

1. **Ouvrir Google Colab** : https://colab.research.google.com/
2. **Créer un nouveau notebook**
3. **Copier-coller les cellules** dans l'ordre
4. **Exécuter séquentiellement** (Shift + Enter)
5. **Attendre la fin de chaque étape** avant de passer à la suivante

---

## ⚡ **Raccourcis Colab**

| Action | Raccourci |
|--------|-----------|
| Exécuter cellule | `Ctrl/Cmd + Enter` |
| Exécuter et passer à suivante | `Shift + Enter` |
| Ajouter cellule | `Ctrl/Cmd + M B` |
| Arrêter exécution | `Ctrl/Cmd + M I` |
| Mode commande | `Esc` |

---

## 🕐 **Temps estimé**

| Étape | Durée (GPU T4) |
|-------|----------------|
| Installation dépendances | 2-3 min |
| Upload dataset | 5-10 min |
| Training SSD | 30-45 min |
| Training YOLOv8n | 40-60 min |
| Training EfficientDet | 50-70 min |
| Training Faster R-CNN | 60-90 min |
| Évaluation multi-modèles | 10-15 min |
| Export modèles | 5-10 min |
| **Total** | **~4-6 heures** |

---

## 💡 **Astuces Colab**

1. **Garder session active** : Exécuter cellule vide périodiquement
2. **Sauvegarder fréquemment** : Copier vers Drive toutes les heures
3. **Limites GPU gratuit** : 12h max, redémarre après
4. **Colab Pro** : GPU plus puissant + 24h runtime

---

## 🔗 **Liens utiles**

- **Repo GitHub** : https://github.com/borisbob91/CropHealth_Detection_PFE
- **Documentation PyTorch** : https://pytorch.org/docs/
- **Ultralytics Docs** : https://docs.ultralytics.com/
- **TensorBoard** : Accessible dans interface Colab

---

<p align="center">
  <strong>🌾 Workflow Colab complet ! Bon training ! 🚀</strong>
</p>