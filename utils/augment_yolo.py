import os
import re
import cv2
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
from collections import Counter
from typing import Dict, List
from tqdm import tqdm

# ============= CONFIGURATION =============

# Nombre d'augmentations par niveau (modifiable)
AUGMENTATION_COUNTS = {
    'low': 18,    # Classes rares → max de diversité
    'medium': 12, # Classes modérées → équilibre
    'high': 6     # Classes fréquentes → minimum suffisant
}

# Seuils de classification des classes
CLASS_THRESHOLDS = {
    'low': 50,
    'medium': 150
}

# ============= TRANSFORMATIONS =============

def get_transforms(dataset_type: str, bbox_format: str = 'yolo') -> Dict[str, A.Compose]:
    """Retourne les transformations selon le dataset_type"""
    bbox_params = A.BboxParams(format=bbox_format, label_fields=['class_labels'], min_visibility=0.3)
    
    all_transforms = {
        'flip_h': {
            'transform': A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'flip_v': {
            'transform': A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'flip_hv': {
            'transform': A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_45': {
            'transform': A.Compose([A.Rotate(limit=(45, 45), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_60': {
            'transform': A.Compose([A.Rotate(limit=(60, 60), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_90': {
            'transform': A.Compose([A.Rotate(limit=(90, 90), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_120': {
            'transform': A.Compose([A.Rotate(limit=(120, 120), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_180': {
            'transform': A.Compose([A.Rotate(limit=(180, 180), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_270': {
            'transform': A.Compose([A.Rotate(limit=(270, 270), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'scale_1': {
            'transform': A.Compose([A.Affine(scale=0.80, p=1.0, mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'scale_2': {
            'transform': A.Compose([A.Affine(scale=1.5, p=1.0, mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': True
        },
        'gamma_1': {
            'transform': A.Compose([A.RandomGamma(gamma_limit=(80, 120), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'gamma_2': {
            'transform': A.Compose([A.RandomGamma(gamma_limit=(150, 200), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'weather_rain': {
            'transform': A.Compose([A.RandomRain(slant_range=(-10, 10), drop_length=30, p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'weather_fog': {
            'transform': A.Compose([A.RandomFog(fog_coef_range=(0.3, 1), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_jitter': {
            'transform': A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
            ], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_hue': {
            'transform': A.Compose([A.HueSaturationValue(hue_shift_limit=20, p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_saturation': {
            'transform': A.Compose([A.HueSaturationValue(sat_shift_limit=(-30, 30), p=1)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
    }
    
    if dataset_type not in ['train', 'val', 'test']:
        raise ValueError(f"dataset_type doit être 'train', 'val' ou 'test'. Reçu : {dataset_type}")
    
    return {
        name: config['transform']
        for name, config in all_transforms.items()
        if config[dataset_type]
    }

# ============= CLASSE PRINCIPALE =============

class YOLOAugmenter:
    """Augmentation YOLO TXT avec équilibrage des classes"""
    
    def __init__(self, input_dir: str, output_dir: str, dataset_type: str = 'train', balance_classes: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_type = dataset_type
        self.balance_classes = balance_classes
        
        # Créer dossiers de sortie
        self.output_images = self.output_dir / 'images'
        self.output_labels = self.output_dir / 'labels'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)
        
        # Charger transformations
        self.transforms = get_transforms(dataset_type)
        
        # Charger classes depuis classes.txt ou notes.json
        self.class_names = self.load_classes()
        
        # Statistiques
        self.stats_before = {}
        self.stats_after = {}
    
    def load_classes(self) -> List[str]:
        """Charge les noms de classes depuis classes.txt ou notes.json"""
        classes_txt = self.input_dir / 'classes.txt'
        notes_json = self.input_dir / 'notes.json'
        
        if classes_txt.exists():
            with open(classes_txt, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        elif notes_json.exists():
            with open(notes_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                categories = sorted(data['categories'], key=lambda x: x['id'])
                return [cat['name'] for cat in categories]
        else:
            print("⚠️ Aucun fichier classes.txt ou notes.json trouvé")
            return []
    
    def auto_detect_patterns(self, filenames: List[str]) -> Dict[str, str]:
        """Détecte les classes depuis les noms de fichiers"""
        pattern_regex = re.compile(r'(?:^|-)([A-Z]+_[A-Z]+_[A-Z]+)(?:_\d+)?')
        file_to_class = {}
        
        for filename in filenames:
            basename = os.path.splitext(filename)[0]
            match = pattern_regex.search(basename)
            if match:
                file_to_class[filename] = match.group(1)
            else:
                file_to_class[filename] = 'unknown'
        
        return file_to_class
    
    def classify_class_level(self, count: int) -> str:
        """Détermine le niveau d'une classe selon son nombre d'images"""
        if count <= CLASS_THRESHOLDS['low']:
            return 'low'
        elif count <= CLASS_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'high'
    
    def parse_yolo_txt(self, txt_path: Path) -> tuple:
        """Parse fichier YOLO TXT"""
        bboxes = []
        class_labels = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_txt(self, txt_path: Path, bboxes: List, class_labels: List):
        """Sauvegarde annotations YOLO"""
        with open(txt_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
    
    def augment(self):
        """Lance l'augmentation avec équilibrage optionnel"""
        images_dir = self.input_dir / 'images'
        labels_dir = self.input_dir / 'labels'
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        # Détection des classes depuis noms de fichiers
        filenames = [f.name for f in image_files]
        file_to_class = self.auto_detect_patterns(filenames)
        
        # Comptage des classes
        class_counter = Counter(file_to_class.values())
        self.stats_before = dict(class_counter)
        
        # Classification par niveau
        class_levels = {cls: self.classify_class_level(count)
                       for cls, count in class_counter.items()}
        
        print(f"\n{'='*60}")
        print(f"🚀 AUGMENTATION YOLO")
        print(f"{'='*60}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Balance classes: {self.balance_classes}")
        print(f"Images trouvées: {len(image_files)}")
        print(f"Classes détectées: {len(class_counter)}")
        print(f"Transformations disponibles: {len(self.transforms)}")
        
        if self.balance_classes:
            print(f"\n📊 Distribution des classes:")
            for level in ['low', 'medium', 'high']:
                classes = [c for c, l in class_levels.items() if l == level]
                if classes:
                    print(f"  {'🔴' if level=='low' else '🟡' if level=='medium' else '🟢'} {level.upper()}: {len(classes)} classes → {AUGMENTATION_COUNTS[level]} augmentations/image")
        
        print(f"\n{'='*60}\n")
        
        # Traitement des images
        augmented_counter = Counter()
        
        for img_path in tqdm(image_files, desc="Augmentation"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            txt_path = labels_dir / (img_path.stem + '.txt')
            if not txt_path.exists():
                continue
            
            bboxes, class_labels = self.parse_yolo_txt(txt_path)
            
            if len(bboxes) == 0:
                continue
            
            # Déterminer le nombre d'augmentations
            img_class = file_to_class.get(img_path.name, 'unknown')
            
            if self.balance_classes:
                if img_class == 'unknown':
                    num_augmentations = len(self.transforms)
                elif img_class in class_levels:
                    level = class_levels[img_class]
                    num_augmentations = AUGMENTATION_COUNTS[level]
                else:
                    num_augmentations = len(self.transforms)
            else:
                num_augmentations = len(self.transforms)
            
            # Sauvegarder original
            cv2.imwrite(
                str(self.output_images / img_path.name),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            self.save_yolo_txt(
                self.output_labels / txt_path.name,
                bboxes,
                class_labels
            )
            augmented_counter[img_class] += 1
            
            # Appliquer transformations
            transform_names = list(self.transforms.keys())[:num_augmentations]
            
            for aug_name in transform_names:
                transform = self.transforms[aug_name]
                
                try:
                    transformed = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    if len(aug_bboxes) == 0:
                        continue
                    
                    new_img_name = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                    new_txt_name = f"{img_path.stem}_{aug_name}.txt"
                    
                    cv2.imwrite(
                        str(self.output_images / new_img_name),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    )
                    
                    self.save_yolo_txt(
                        self.output_labels / new_txt_name,
                        aug_bboxes,
                        aug_labels
                    )
                    augmented_counter[img_class] += 1
                
                except Exception as e:
                    print(f"❌ Erreur {aug_name} sur {img_path.name}: {str(e)}")
        
        self.stats_after = dict(augmented_counter)
        
        # Copier métadonnées
        self.copy_metadata()
        
        total_images = len(list(self.output_images.glob('*')))
        print(f"\n✅ Augmentation terminée!")
        print(f"   Images originales: {len(image_files)}")
        print(f"   Images totales: {total_images}")
        print(f"   Ratio: x{total_images / len(image_files):.2f}")
    
    def copy_metadata(self):
        """Copie classes.txt et notes.json vers output"""
        for filename in ['classes.txt', 'notes.json']:
            src = self.input_dir / filename
            if src.exists():
                shutil.copy(str(src), str(self.output_dir / filename))
                print(f"📄 Copie: {filename}")
    
    def export_statistics(self):
        """Exporte les statistiques en CSV"""
        classes = sorted(set(list(self.stats_before.keys()) + list(self.stats_after.keys())))
        
        data = []
        for cls in classes:
            before = self.stats_before.get(cls, 0)
            after = self.stats_after.get(cls, 0)
            level = self.classify_class_level(before)
            
            data.append({
                'Classe': cls,
                'Niveau': level.upper(),
                'Images_Avant': before,
                'Images_Après': after,
                'Augmentation': after - before,
                'Ratio': f"x{after/before:.2f}" if before > 0 else "N/A"
            })
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_dir / 'statistics.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n📊 Statistiques sauvegardées: {csv_path}")
        
        return df

def plot_statistics(df: pd.DataFrame, output_dir: Path):
    """Génère les graphiques de statistiques"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique empilé
    ax1 = axes[0]
    classes = df['Classe'].values
    before = df['Images_Avant'].values
    after_only = df['Augmentation'].values
    
    x = np.arange(len(classes))
    width = 0.6
    
    ax1.bar(x, before, width, label='Original', color='#3498db', alpha=0.8)
    ax1.bar(x, after_only, width, bottom=before, label='Augmenté', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nombre d\'images', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Avant/Après (Empilé)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Graphique côte à côte
    ax2 = axes[1]
    x = np.arange(len(classes))
    width = 0.35
    
    ax2.bar(x - width/2, before, width, label='Avant', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, df['Images_Après'].values, width, label='Après', color='#2ecc71', alpha=0.8)
    
    ax2.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nombre d\'images', fontsize=12, fontweight='bold')
    ax2.set_title('Comparaison Avant/Après (Côte à côte)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'statistics_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Graphiques sauvegardés: {plot_path}")
    
    plt.show()

# ============= UTILISATION =============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Augmentation YOLO TXT avec équilibrage des classes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python augment_yolo.py
  python augment_yolo.py --input ./yolo_dataset --output ./augmented
  python augment_yolo.py --dataset_type val --balance False
  python augment_yolo.py -i ./data -o ./output -t train -b True
# Google Colab
!python voc_to_yolo.py \
    --input /content/drive/MyDrive/dataset_voc \
    --output /content/drive/MyDrive/dataset_yolo
    
```
Structure attendue (input):
  input_dir/
  ├── images/
  ├── labels/
  ├── classes.txt (ou notes.json)
  └── notes.json (optionnel)
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=r'C:\Users\BorisBob\Desktop\collecte\test\project-3-at-2025-11-14-17-04-4ff9840d\yolo_dataset',
        help='Dossier d\'entrée contenant images/, labels/ et classes.txt'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=r'C:\Users\BorisBob\Desktop\collecte\test\project-3-at-2025-11-14-17-04-4ff9840d\yolo_augmented',
        help='Dossier de sortie pour les données augmentées'
    )
    
    parser.add_argument(
        '--dataset_type', '-t',
        type=str,
        choices=['train', 'val', 'test'],
        default='train',
        help='Type de dataset (train/val/test) - détermine les transformations appliquées'
    )
    
    parser.add_argument(
        '--balance', '-b',
        type=lambda x: str(x).lower() == 'true',
        default=True,
        help='Activer l\'équilibrage des classes (True/False)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔧 CONFIGURATION")
    print("="*60)
    print(f"Input dir:      {args.input}")
    print(f"Output dir:     {args.output}")
    print(f"Dataset type:   {args.dataset_type}")
    print(f"Balance:        {args.balance}")
    print("="*60 + "\n")
    
    augmenter = YOLOAugmenter(
        input_dir=args.input,
        output_dir=args.output,
        dataset_type=args.dataset_type,
        balance_classes=args.balance
    )
    
    augmenter.augment()
    
    df = augmenter.export_statistics()
    
    plot_statistics(df, Path(args.output))
    
    print("\n🎉 Pipeline terminé avec succès!")