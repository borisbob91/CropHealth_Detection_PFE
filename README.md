[![Open porject In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisbob91/CropHealth_Detection_PFE/blob/main/notebooks/complete_workflow.ipynb)
[![Setup In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisbob91/CropHealth_Detection_PFE/blob/main/notebooks/setup_colab.ipynb)
[![train models In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisbob91/CropHealth_Detection_PFE/blob/main/notebooks/train_model.ipynb)
[![Evaluate In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisbob91/CropHealth_Detection_PFE/blob/main/notebooks/evaluation.ipynb)
# 🌾 CropHealth Detection - Projet de Fin d'Études (PFE)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Détection et classification d'insectes nuisibles, auxiliaires et symptômes foliaires sur plants de coton** via deep learning pour l'agriculture de précision.

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Architectures](#-architectures-implémentées)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Utilisation](#-utilisation)
- [Évaluation](#-évaluation)
- [Résultats](#-résultats)
- [Références](#-références)
- [Auteur](#-auteur)

---

## 🎯 À propos

Ce projet implémente et compare **5 architectures de détection d'objets** pour identifier automatiquement les ravageurs, insectes auxiliaires et maladies foliaires sur images de plants de coton. L'objectif est de déployer ces modèles sur appareils embarqués (drones, smartphones) pour assister les agriculteurs dans la surveillance des cultures.

### 🔬 Contexte scientifique

Les modèles ont été sélectionnés selon leur **validation en agriculture** :
- **SSD MobileNetV3** : Saleem et al. (2019) - Détection maladies foliaires
- **YOLOv8n** : Zhang et al. (2025), Wei et al. (2024) - Ravageurs du coton
- **EfficientDet-D0** : Tan et al. (2020) - Compromis précision/vitesse
- **Faster R-CNN** : Fuentes et al. (2017) - Maladies de la tomate (mAP > 85%)

### 📊 Métriques d'évaluation

Conformément aux protocoles standard (Padilla et al., 2020) :
- **mAP@50** (principal) : via COCO Evaluator (PyTorch) / Ultralytics
- **AP@50 par classe** : via `torchmetrics.detection.MeanAveragePrecision`
- **F1-Score** : Precision/Recall (micro average)
- **FPS / GFLOPs / Taille** : Contraintes edge deployment

---

## 🏗️ Architectures implémentées

| Architecture | Type | Backbone | Input | Params | GFLOPs | Taille | FPS | Validation |
|-------------|------|----------|-------|--------|--------|--------|-----|------------|
| **SSD** | One-Stage | MobileNetV3 | 320×320 | 5.5M | 1.2 | 22 MB | 60 | Saleem et al. (2019) |
| **YOLOv8n** | One-Stage | CSP-Darknet | 640×640 | 3.15M | 8.7 | 6 MB | 120 | Zhang et al. (2025) |
| **EfficientDet-D0** | One-Stage | EfficientNet-B0 | 512×512 | 3.9M | 2.5 | 16 MB | 90 | Tan et al. (2020) |
| **Faster R-CNN** | Two-Stage | ResNet50 + FPN | 800×800 | 43.7M | 280 | 175 MB | 10 | Fuentes et al. (2017) |
| **Faster R-CNN (light)** | Two-Stage | MobileNetV3 + FPN | 320×320 | 6.0M | 15 | 25 MB | 25-30 | Fuentes et al. (2017) |

---

## 🚀 Installation

### Prérequis

- **Python** : 3.8 - 3.10 (3.11 compatible, 3.12 non supporté)
- **CUDA** : 11.8+ (optionnel, CPU supporté)
- **GPU** : 8+ GB VRAM recommandé

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/borisbob91/CropHealth_Detection_PFE.git
cd CropHealth_Detection_PFE

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer dépendances
pip install -r requirements.txt

# Vérifier GPU (optionnel)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Installation Google Colab

```python
# Cloner et installer
!git clone https://github.com/borisbob91/CropHealth_Detection_PFE.git
%cd CropHealth_Detection_PFE
!pip install -q -r requirements.txt

# OU utiliser le script setup
!wget https://raw.githubusercontent.com/borisbob91/CropHealth_Detection_PFE/main/setup_notebook.py
%run setup_notebook.py --install-deps
```

---

## 📁 Structure du projet

```
CropHealth_Detection_PFE/
├── configs/
│   └── model_configs.py          # Hyperparamètres centralisés
├── datasets/
│   ├── transforms.py             # Augmentations Albumentations
│   ├── yolo_dataset.py           # Dataset YOLO txt
│   └── coco_dataset.py           # Dataset COCO JSON
├── models/
│   ├── ssd_model.py              # SSD MobileNetV3
│   ├── efficientdet_model.py    # EfficientDet-D0
│   ├── fasterrcnn_model.py      # Faster R-CNN ResNet50
│   └── fasterrcnn_light_model.py # Faster R-CNN MobileNetV3
├── trainers/
│   └── base_trainer.py           # Boucle train/eval unifiée
├── utils/
│   ├── yolo2coco.py              # Conversion YOLO → COCO
│   └── yolo_wrapper.py           # Wrapper YOLOv8n pour évaluation
├── notebooks/
│   └── colab_quickstart_cells.md # Template Google Colab
├── train.py                      # CLI training (SSD, EfficientDet, Faster R-CNN)
├── train_yolo.py                 # CLI training YOLOv8n
├── predict.py                    # Inférence + métriques (COCO Evaluator)
├── yolo_predict.py               # Inférence YOLOv8n (Ultralytics)
├── predict_classification.py    # Inférence modèles classification
├── evaluate_models.py            # Évaluation multi-modèles (AP@50, F1, PR curves, CM)
├── setup_notebook.py             # Setup automatique Colab/Jupyter
├── requirements.txt              # Dépendances production
├── requirements-dev.txt          # Dépendances développement
└── README.md
```

---

## 🎓 Utilisation

### 1️⃣ Training

#### **SSD MobileNetV3**
```bash
python train.py --model ssd --data data/yolo_crop --device cuda
```

#### **YOLOv8n**
```bash
python train_yolo.py --data data/yolo_crop/data.yaml --device 0 --cache
```

#### **EfficientDet-D0** (nécessite conversion COCO)
```bash
# Conversion YOLO → COCO
python utils/yolo2coco.py --yolo-root data/yolo_crop --output data/coco_crop

# Training
python train.py --model efficientdet --data data/coco_crop --device cuda
```

#### **Faster R-CNN (ResNet50 / MobileNetV3)**
```bash
# Version classique (ResNet50)
python train.py --model fasterrcnn --data data/yolo_crop --device cuda

# Version light (MobileNetV3)
python train.py --model fasterrcnn_light --data data/yolo_crop --device cuda
```

### 2️⃣ Inférence

#### **Prédictions + Visualisations**
```bash
python predict.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --input data/test/images \
    --output predictions/ssd \
    --conf 0.5
```

#### **Prédictions + Métriques complètes**
```bash
python predict.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --input data/test/images \
    --val-data data/yolo_crop \
    --output predictions/ssd
```

### 3️⃣ Évaluation multi-modèles

```bash
python evaluate_models.py \
    --checkpoints \
        ssd:runs/CropHealth_SSD_1107_1430/best.pt \
        yolov8n:runs/CropHealth_YOLOv8n_1107_1445/weights/best.pt \
        fasterrcnn:runs/CropHealth_FasterRCNN_1107_1515/best.pt \
    --val-data data/yolo_crop \
    --output evaluation_results \
    --device cuda
```

**Outputs générés** :
- `ap50_per_class.csv` : AP@50 par classe (tous modèles)
- `f1_per_class.csv` : F1-Score par classe
- `global_metrics.csv` : mAP@50 + F1 micro
- `ap50_per_class_comparison.png` : Bar plot comparatif
- `f1_per_class_comparison.png` : Bar plot F1-Score
- `pr_curve_class_X.png` : Courbes Precision-Recall
- `confusion_matrix_MODEL.png` : Matrices de confusion

---

## 📊 Évaluation

### Structure dataset attendue

#### **Format YOLO** (SSD, Faster R-CNN, YOLOv8n)
```
data/yolo_crop/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt  # cls xc yc w h (normalized 0-1)
│       └── img2.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml  # Pour YOLOv8n uniquement
```

**data.yaml** (YOLOv8n) :
```yaml
path: /path/to/data/yolo_crop
train: train/images
val: val/images
nc: 9
names: ['class_0', 'class_1', ..., 'class_8']
```

#### **Format COCO JSON** (EfficientDet-D0)
```
data/coco_crop/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

### Métriques calculées

| Métrique | Rôle | Outil |
|----------|------|-------|
| **mAP@50** | Principal | COCO Evaluator / Ultralytics |
| **mAP@50-95** | Secondaire | COCO Evaluator / Ultralytics |
| **AP@50 par classe** | Analyse performance | `torchmetrics.MeanAveragePrecision` |
| **Precision** | Analyse FP | COCO Evaluator |
| **Recall** | Analyse FN | COCO Evaluator |
| **F1-Score (micro)** | Global | Calcul TP/FP/FN |
| **F1-Score par classe** | Analyse classes | Calcul TP/FP/FN |
| **FPS** | Temps-réel | `time.perf_counter()` |
| **Taille / GFLOPs** | Edge deployment | `torchinfo`, `thop` |

---

## 🏆 Résultats

### Exemple de résultats 

| Modèle | mAP@50 (%) | F1-Score (%) | FPS | Taille (MB) | GFLOPs |
|--------|------------|--------------|-----|-------------|--------|
| **SSD** | 88.99 | 89.06 | 60 | 22 | 1.2 |
| **YOLOv8n** | 92.18 | 91.45 | 120 | 6 | 8.7 |
| **EfficientDet-D0** | 89.14 | 88.67 | 90 | 16 | 2.5 |
| **Faster R-CNN** | 93.32 | 92.33 | 10 | 175 | 280 |
| **Faster R-CNN (light)** | 87.61 | 86.78 | 30 | 25 | 15 |

### Visualisations

![AP@50 par classe](docs/images/ap50_per_class_comparison.png)
*Figure 1 : Comparaison AP@50 par classe pour tous les modèles*

![Courbes Precision-Recall](docs/images/pr_curve_example.png)
*Figure 2 : Courbes Precision-Recall (exemple classe_0)*

![Confusion Matrix](docs/images/confusion_matrix_example.png)
*Figure 3 : Matrice de confusion (exemple SSD)*

---

## 📚 Références

### Architectures

1. **Saleem, M. H., et al.** (2019). Plant disease detection and classification by deep learning. *Plants*, 8(11), 468.

2. **Zhang, Y., et al.** (2025). Snake-YOLO: An Improved YOLOv8 for Cotton Pests Detection. *IEEE Access*.

3. **Wei, X., et al.** (2024). Improved YOLOv8 for cotton pests and diseases detection. *Computers and Electronics in Agriculture*.

4. **Fuentes, A., et al.** (2017). A robust deep-learning-based detector for real-time tomato plant diseases and pests recognition. *Sensors*, 17(9), 2022.

5. **Tan, M., Pang, R., & Le, Q. V.** (2020). EfficientDet: Scalable and efficient object detection. *CVPR 2020*.

### Métriques

6. **Padilla, R., et al.** (2020). A survey on performance metrics for object-detection algorithms. *International Conference on Systems, Signals and Image Processing*.

7. **Khan, A., et al.** (2025). Recent advances in deep learning-based plant disease detection: A systematic review. *Computers and Electronics in Agriculture*.

8. **Raza, A., et al.** (2025). A comprehensive review of deep learning methods for agricultural applications. *IEEE Access*.

### Frameworks

- **PyTorch** : https://pytorch.org/
- **Ultralytics YOLOv8** : https://github.com/ultralytics/ultralytics
- **EfficientDet (rwightman)** : https://github.com/rwightman/efficientdet-pytorch
- **timm** : https://github.com/huggingface/pytorch-image-models

---

## 👨‍💻 Auteur

**Boris Bob KENGNE DJEUTANE**  
Projet de Fin d'Études (PFE) - Détection d'objets pour l'agriculture de précision

- 📧 Email : [votre.email@example.com](mailto:votre.email@example.com)
- 🔗 GitHub : [@borisbob91](https://github.com/borisbob91)
- 🌐 LinkedIn : [Votre profil LinkedIn](https://linkedin.com/in/votre-profil)

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Encadrants** : [Noms des encadrants]
- **Institution** : [Nom de votre université/école]
- **Communautés** : PyTorch, Ultralytics, OpenCV, Albumentations

---

## 📝 Citation

Si vous utilisez ce travail, veuillez citer :

```bibtex
@mastersthesis{kouacouboris2025crophealth,
  title={CropHealth Detection: Deep Learning pour la détection automatique d'insectes et maladies du coton},
  author={KOUACOU GHISLAIN BORIS, BorisBob},
  year={2025},
  school={[INP-HB]},
  type={Projet de Fin d'Études},
  url={https://github.com/borisbob91/CropHealth_Detection_PFE}
}
```

---

## 🐛 Problèmes connus

- **Windows** : `pycocotools` nécessite Visual Studio Build Tools ([télécharger ici](https://visualstudio.microsoft.com/downloads/))
- **macOS M1/M2** : PyTorch MPS backend en beta, privilégier CPU
- **CUDA OOM** : Réduire `batch_size` dans `configs/model_configs.py`

---

## 🔮 Travaux futurs

- [ ] Support YOLOv11
- [ ] Quantization INT8 pour déploiement mobile
- [ ] Export ONNX / TensorRT
- [ ] Interface web Gradio/Streamlit
- [ ] API REST Flask/FastAPI
- [ ] Dataset augmenté (GAN, mixup)

---

<p align="center">
  <strong>⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for sustainable agriculture 🌾
</p>