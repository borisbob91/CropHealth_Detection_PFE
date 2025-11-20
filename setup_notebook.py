#!/usr/bin/env python3
"""
CropHealth Detection - Notebook Setup Script
Prépare l'environnement pour Google Colab / Jupyter / Kaggle

Usage dans un notebook:
    !wget https://raw.githubusercontent.com/YOUR_REPO/setup_notebook.py
    %run setup_notebook.py --install-deps
    
Ou si projet déjà cloné:
    %run setup_notebook.py --install-deps
"""
import os
import sys
import subprocess
from pathlib import Path


def check_environment():
    """Détecte l'environnement (Colab / Kaggle / Local)"""
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    else:
        return 'local'


def install_dependencies():
    """Installe les dépendances PyTorch"""
    env = check_environment()
    print(f"\n{'='*60}")
    print(f"🔧 Environment: {env.upper()}")
    print(f"{'='*60}\n")
    
    packages = [
        'torch',
        'torchvision',
        'torchmetrics',
        'albumentations',
        'ultralytics',
        'effdet',
        'timm',
        'tensorboard',
    ]
    
    print("📦 Installing dependencies...")
    for pkg in packages:
        print(f"  - {pkg}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
    
    print("\n✅ Dependencies installed!")


def setup_project_structure():
    """Crée la structure de dossiers si nécessaire"""
    dirs = [
        'configs',
        'datasets',
        'models',
        'trainers',
        'utils',
        'runs',
        'data',
        'notebooks',
    ]
    
    print(f"\n📁 Setting up project structure...")
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✓ {d}/")
    
    print("\n✅ Project structure ready!")


def check_gpu():
    """Vérifie disponibilité GPU"""
    import torch
    
    print(f"\n🖥️  GPU Check:")
    if torch.cuda.is_available():
        print(f"  ✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ CUDA Version: {torch.version.cuda}")
        print(f"  ✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠️  No GPU detected - using CPU")


def download_project_files():
    """Télécharge les fichiers du projet depuis GitHub (optionnel)"""
    print(f"\n📥 To download project files from GitHub:")
    print(f"  !git clone https://github.com/YOUR_USERNAME/CropHealth_Detection.git")
    print(f"  %cd CropHealth_Detection")


def print_usage_examples():
    """Affiche exemples d'utilisation"""
    print(f"\n{'='*60}")
    print(f"📚 USAGE EXAMPLES")
    print(f"{'='*60}\n")
    
    examples = [
        ("SSD MobileNetV3", "!python train.py --model ssd --data /content/data/yolo_crop --device cuda"),
        ("YOLOv8n", "!python train_yolo.py --data /content/data/yolo_crop/data.yaml --device 0"),
        ("EfficientDet-D0", "!python utils/yolo2coco.py --yolo-root /content/data/yolo_crop --output /content/data/coco_crop\n!python train.py --model efficientdet --data /content/data/coco_crop --device cuda"),
        ("Faster R-CNN", "!python train.py --model fasterrcnn --data /content/data/yolo_crop --device cuda"),
        ("Faster R-CNN Light", "!python train.py --model fasterrcnn_light --data /content/data/yolo_crop --device cuda"),
    ]
    
    for name, cmd in examples:
        print(f"🔹 {name}:")
        print(f"  {cmd}\n")
    
    print(f"📊 View TensorBoard:")
    print(f"  %load_ext tensorboard")
    print(f"  %tensorboard --logdir runs/\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CropHealth Notebook Setup')
    parser.add_argument('--install-deps', action='store_true',
                        help='Install Python dependencies')
    parser.add_argument('--skip-examples', action='store_true',
                        help='Skip usage examples')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Notebook Setup")
    print(f"{'='*60}")
    
    # Check environment
    env = check_environment()
    print(f"\n📍 Environment: {env.upper()}")
    
    # Install dependencies
    if args.install_deps:
        install_dependencies()
    else:
        print(f"\n⏭️  Skipping dependency installation (use --install-deps to install)")
    
    # Setup structure
    setup_project_structure()
    
    # Check GPU
    if args.install_deps:  # Only check if torch installed
        check_gpu()
    
    # Usage examples
    if not args.skip_examples:
        print_usage_examples()
    
    print(f"\n{'='*60}")
    print(f"✅ Setup complete! Ready to train.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()