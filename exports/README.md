# 📦 CropHealth Detection - Model Export

Export des modèles entraînés vers **ONNX**, **TensorRT**, **TFLite (INT8)**, **CoreML** pour déploiement mobile et edge.

---

## 🎯 Formats supportés

| Format | Plateforme | Quantization | Accélération | Usage |
|--------|-----------|--------------|--------------|-------|
| **ONNX** | Multi-plateforme | FP32, FP16 | CPU, GPU | Interopérabilité |
| **TensorRT** | NVIDIA GPU | FP32, FP16, INT8 | GPU | Déploiement GPU haute performance |
| **TFLite** | Android, iOS, Edge | FP32, FP16, INT8 | CPU, GPU, NPU | Mobile/Edge devices |
| **CoreML** | iOS, macOS | FP32, FP16 | Apple Neural Engine | iPhone, iPad, Mac |

---

## 🚀 Installation dépendances

```bash
# Export ONNX
pip install onnx onnx-simplifier

# Export TensorRT (NVIDIA GPU requis)
pip install nvidia-tensorrt

# Export TFLite
pip install tensorflow onnx-tf

# Export CoreML (macOS recommandé)
pip install coremltools

# Benchmark
pip install onnxruntime onnxruntime-gpu
```

---

## 📋 Utilisation

### 1️⃣ Export ONNX

```bash
# SSD MobileNetV3
python export/export_models.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --format onnx \
    --output exports/ssd

# Faster R-CNN
python export/export_models.py \
    --model fasterrcnn \
    --checkpoint runs/CropHealth_FasterRCNN_1107_1515/best.pt \
    --format onnx \
    --output exports/fasterrcnn
```

**Output** : `exports/ssd/CropHealth_SSD.onnx`

---

### 2️⃣ Export TensorRT (FP16)

```bash
# Étape 1: Export ONNX
python export/export_models.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --format onnx \
    --output exports/ssd

# Étape 2: Convert ONNX → TensorRT
python export/export_models.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --format tensorrt \
    --quantize float16 \
    --output exports/ssd
```

**Output** : `exports/ssd/CropHealth_SSD.engine`

---

### 3️⃣ Export TFLite INT8 (Quantization)

```bash
python export/export_models.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --format tflite \
    --quantize int8 \
    --calibration-data data/yolo_crop/train/images \
    --output exports/ssd
```

**Output** : `exports/ssd/CropHealth_SSD_int8.tflite`

**Note** : INT8 nécessite ~100 images de calibration pour quantization post-training.

---

### 4️⃣ Export CoreML (iOS/macOS)

```bash
python export/export_models.py \
    --model ssd \
    --checkpoint runs/CropHealth_SSD_1107_1430/best.pt \
    --format coreml \
    --output exports/ssd
```

**Output** : `exports/ssd/CropHealth_SSD.mlpackage`

---

### 5️⃣ Export YOLOv8n (via Ultralytics)

```bash
# ONNX
python export/export_models.py \
    --model yolov8n \
    --checkpoint runs/CropHealth_YOLOv8n_1107_1445/weights/best.pt \
    --format onnx \
    --output exports/yolov8n

# TFLite INT8
python export/export_models.py \
    --model yolov8n \
    --checkpoint runs/CropHealth_YOLOv8n_1107_1445/weights/best.pt \
    --format tflite \
    --quantize int8 \
    --calibration-data data/yolo_crop/data.yaml \
    --output exports/yolov8n

# TensorRT FP16
python export/export_models.py \
    --model yolov8n \
    --checkpoint runs/CropHealth_YOLOv8n_1107_1445/weights/best.pt \
    --format tensorrt \
    --quantize float16 \
    --output exports/yolov8n
```

---

## 📊 Benchmark des exports

Comparez PyTorch, ONNX, TensorRT, TFLite :

```bash
python export/benchmark_exports.py \
    --model ssd \
    --pytorch runs/CropHealth_SSD_1107_1430/best.pt \
    --onnx exports/ssd/CropHealth_SSD.onnx \
    --tensorrt exports/ssd/CropHealth_SSD.engine \
    --tflite exports/ssd/CropHealth_SSD_int8.tflite \
    --runs 100 \
    --output benchmark_ssd.csv
```

**Exemple output** :

```
Backend              Time (ms)       FPS        Device    
-------------------------------------------------------
PyTorch              16.54           60.46      cuda      
ONNX Runtime         12.32           81.17      CUDA      
TensorRT             5.23            191.20     CUDA      
TFLite               45.67           21.90      CPU       
```

**Speedup** :
- TensorRT FP16 : ~3.2x vs PyTorch
- ONNX Runtime : ~1.3x vs PyTorch
- TFLite INT8 : Optimal pour mobile (taille réduite)

---

## 🎯 Recommandations par plateforme

### **NVIDIA Jetson** (Nano, Xavier, Orin)
- **Format** : TensorRT FP16
- **Modèle** : SSD MobileNetV3 ou YOLOv8n
- **FPS attendu** : 30-60 FPS (Jetson Orin)

```bash
python export/export_models.py --model ssd --format tensorrt --quantize float16
```

### **Raspberry Pi 4/5**
- **Format** : TFLite INT8
- **Modèle** : SSD MobileNetV3 (le plus léger)
- **FPS attendu** : 5-10 FPS

```bash
python export/export_models.py --model ssd --format tflite --quantize int8
```

### **Android (smartphone)**
- **Format** : TFLite INT8 + GPU delegate
- **Modèle** : YOLOv8n ou SSD
- **FPS attendu** : 15-30 FPS (flagship)

```bash
python export/export_models.py --model yolov8n --format tflite --quantize int8
```

### **iPhone/iPad**
- **Format** : CoreML
- **Modèle** : EfficientDet-D0 ou YOLOv8n
- **FPS attendu** : 30-60 FPS (Apple Neural Engine)

```bash
python export/export_models.py --model efficientdet --format coreml
```

---

## 📐 Tailles des modèles exportés

| Modèle | PyTorch (FP32) | ONNX (FP32) | TensorRT (FP16) | TFLite (INT8) | CoreML |
|--------|----------------|-------------|-----------------|---------------|---------|
| **SSD** | 22 MB | 22 MB | 11 MB | 6 MB | 23 MB |
| **YOLOv8n** | 6 MB | 6 MB | 3 MB | 2 MB | 7 MB |
| **EfficientDet-D0** | 16 MB | 16 MB | 8 MB | 4 MB | 17 MB |
| **Faster R-CNN** | 175 MB | 175 MB | 88 MB | 45 MB | 180 MB |
| **Faster R-CNN (light)** | 25 MB | 25 MB | 13 MB | 7 MB | 27 MB |

---

## ⚙️ Optimisations avancées

### INT8 Quantization (Post-Training)

**Avantages** :
- 4x réduction taille
- 2-4x speedup sur CPU/NPU
- Perte précision minime (<2% mAP)

**Calibration** :
- 100-500 images représentatives
- Distribution similaire au dataset de training

```bash
# Utiliser subset du train set
python export/export_models.py \
    --format tflite \
    --quantize int8 \
    --calibration-data data/yolo_crop/train/images
```

### ONNX Graph Simplification

Réduit le graphe ONNX pour meilleur runtime :

```bash
# Automatique avec --simplify dans export_models.py
pip install onnx-simplifier
python -m onnxsim input.onnx output_simplified.onnx
```

### TensorRT INT8 Calibration

```python
# Nécessite implémentation custom calibrator
# Voir: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8
```

---

## 🐛 Problèmes connus

### **ONNX export échoue**
```bash
# Solution: Downgrade opset version
python export/export_models.py --format onnx --opset-version 11
```

### **TensorRT incompatible avec certaines ops**
```bash
# Solution: Fallback vers ONNX Runtime
python export/export_models.py --format onnx
# Utiliser onnxruntime-gpu pour inférence
```

### **TFLite quantization échoue**
```bash
# Solution: Utiliser FP16 au lieu de INT8
python export/export_models.py --format tflite --quantize float16
```

### **CoreML export erreur (non-macOS)**
```bash
# Solution: Exporter sur macOS ou utiliser Docker
docker run -it --rm -v $(pwd):/workspace python:3.10
pip install coremltools torch torchvision
```

---

## 📚 Ressources

- [ONNX Documentation](https://onnx.ai/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TFLite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)
- [CoreML Tools](https://coremltools.readme.io/)
- [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export/)

---

## ✅ Checklist export production

- [ ] Exporter en ONNX (validation inter-framework)
- [ ] Benchmark ONNX vs PyTorch (vérifier précision)
- [ ] Exporter format cible (TensorRT/TFLite/CoreML)
- [ ] Tester inférence sur device cible
- [ ] Benchmark FPS réel sur device
- [ ] Valider mAP@50 après quantization (<2% drop)
- [ ] Tester edge cases (faible luminosité, occlusion)
- [ ] Packager modèle + metadata (classes, input size)

---

<p align="center">
  <strong>📦 Ready for deployment! 🚀</strong>
</p>