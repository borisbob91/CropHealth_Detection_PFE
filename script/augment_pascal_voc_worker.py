import os
import random
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import albumentations as A
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ============= CONFIGURATION =============

# Nombre d'augmentations par niveau (modifiable)
AUGMENTATION_COUNTS = {
    'low': 13,    # Classes rares → max de diversité
    'medium': 7, # Classes modérées → équilibre
    'high': 5    # Classes fréquentes → minimum suffisant
}

# Seuils de classification des classes
CLASS_THRESHOLDS = {
    'low': 150,
    'medium': 500
}


# ============= TRANSFORMATIONS =============

def get_transforms(dataset_type: str, bbox_format: str = 'pascal_voc') -> Dict[str, A.Compose]:
    """
    Retourne UNIQUEMENT les transformations activées pour le dataset_type donné.
    
    Args:
        dataset_type (str): 'train', 'val' ou 'test'
        bbox_format (str): format des bbox ('pascal_voc', 'coco', etc.)
    
    Returns:
        Dict[str, A.Compose]: seulement les transformations où {dataset_type}: True
    """
    if dataset_type not in ['train', 'val', 'test']:
        raise ValueError("dataset_type doit être 'train', 'val' ou 'test'")

    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=['class_labels'],
        min_area=10,
        min_visibility=0.2
    )

    # === Toutes les transformations avec leur activation par split ===
    all_transforms = {
        'flip_h': {
            'transform': A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'flip_v': {
            'transform': A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'rot_270': {
            'transform': A.Compose([A.Rotate(limit=(270, 270), p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'weather_rain': {
            'transform': A.Compose([A.RandomRain(slant_range=(-10, 10), drop_length=30, p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
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
        'flip_hv': {
            'transform': A.Compose([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], bbox_params=bbox_params),
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
        'scale': {
            'transform': A.Compose([A.Affine(scale=1.2, p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': True, 'test': True
        },
        'gamma': {
            'transform': A.Compose([A.RandomGamma(gamma_limit=(150, 200), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_jitter': {
            'transform': A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
            ], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
           'weather_fog': {
            'transform': A.Compose([A.RandomFog(fog_coef_range=(0.3, 1), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_hue': {
            'transform': A.Compose([A.HueSaturationValue(hue_shift_limit=20, p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'color_saturation': {
            'transform': A.Compose([A.HueSaturationValue(sat_shift_limit=(-30, 30), p=1.0)], bbox_params=bbox_params),
            'train': True, 'val': False, 'test': False
        },
        'noise_gauss': {
        'transform': A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)], bbox_params=bbox_params),
        'train': True, 'val': False, 'test': False
    },
    'shift_scl_rot': {
    'transform': A.Compose([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbox_params),
    'train': True, 'val': False, 'test': False
    },
    'blur_gaussian': {
        'transform': A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)], bbox_params=bbox_params),
        'train': True, 'val': False, 'test': False
    },
    }

    # === Filtrage selon le dataset_type ===
    filtered_transforms = {}
    for name, config in all_transforms.items():
        if config.get(dataset_type, False):  # si activé pour ce split
            filtered_transforms[name] = config['transform']

    print(f"→ {len(filtered_transforms)} transformations chargées pour '{dataset_type}'")
    return filtered_transforms


# ============= FONCTION WORKER POUR MULTIPROCESSING =============

def process_single_image_worker(args):
    """Worker pour traiter une image (pour multiprocessing)"""
    (img_path, labels_dir, output_images, output_annots, 
     transforms_dict, file_to_class, class_levels, 
     balance_classes, dataset_type, num_augmentations_per_class) = args
    
    augmented_counter = Counter()
    
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return augmented_counter
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        xml_path = labels_dir / (img_path.stem + '.xml')
        if not xml_path.exists():
            return augmented_counter
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        bboxes = []
        class_labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            bboxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(class_name)
        
        size = root.find('size')
        orig_width = int(size.find('width').text)
        orig_height = int(size.find('height').text)
        
        if len(bboxes) == 0:
            return augmented_counter
        
        # Déterminer le nombre d'augmentations
        img_class = file_to_class.get(img_path.name, 'unknown')
        
        if balance_classes:
            if img_class == 'unknown':
                num_augmentations = len(transforms_dict)
            elif img_class in class_levels:
                level = class_levels[img_class]
                num_augmentations = num_augmentations_per_class[level]
            else:
                num_augmentations = len(transforms_dict)
        else:
            num_augmentations = len(transforms_dict)
        
        # Sauvegarder original
        cv2.imwrite(
            str(output_images / img_path.name),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        
        # Save XML
        save_pascal_voc_xml(
            output_annots / xml_path.name,
            img_path.name,
            bboxes,
            class_labels,
            orig_width,
            orig_height
        )
        augmented_counter[img_class] += 1
        
        # Appliquer transformations
        transform_names = list(transforms_dict.keys())
        random.shuffle(transform_names)
        transform_names = random.sample(transform_names, k=min(num_augmentations, len(transform_names)))
        
        for aug_name in transform_names:
            transform = transforms_dict[aug_name]
            
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
                new_xml_name = f"{img_path.stem}_{aug_name}.xml"
                
                cv2.imwrite(
                    str(output_images / new_img_name),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                
                new_height, new_width = aug_image.shape[:2]
                save_pascal_voc_xml(
                    output_annots / new_xml_name,
                    new_img_name,
                    aug_bboxes,
                    aug_labels,
                    new_width,
                    new_height
                )
                augmented_counter[img_class] += 1
            
            except Exception as e:
                pass  # Silencieux pour multiprocessing
        
        return augmented_counter
    
    except Exception as e:
        return augmented_counter


def save_pascal_voc_xml(xml_path: Path, image_name: str, bboxes: List, 
                       class_labels: List, img_width: int, img_height: int, img_depth: int = 3):
    """Sauvegarde fichier Pascal VOC XML"""
    annotation = ET.Element('annotation')
    
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'
    
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name
    
    path = ET.SubElement(annotation, 'path')
    path.text = str(Path('images') / image_name)
    
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_width)
    height = ET.SubElement(size, 'height')
    height.text = str(img_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img_depth)
    
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    for bbox, class_name in zip(bboxes, class_labels):
        obj = ET.SubElement(annotation, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = str(class_name)
        
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(bbox[0]))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(bbox[1]))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(bbox[2]))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(bbox[3]))
    
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space='  ')
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


# ============= CLASSE PRINCIPALE =============

class PascalVOCAugmenter:
    """Augmentation Pascal VOC XML avec équilibrage des classes"""
    
    def __init__(self, input_dir: str, output_dir: str, dataset_type: str = 'train', 
                 balance_classes: bool = True, num_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_type = dataset_type
        self.balance_classes = balance_classes
        self.num_workers = num_workers
        
        # Créer dossiers de sortie
        self.output_images = self.output_dir / 'images'
        self.output_annots = self.output_dir / 'Annotations'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_annots.mkdir(parents=True, exist_ok=True)
        
        # Charger transformations
        self.transforms = get_transforms(dataset_type)
        
        # Statistiques
        self.stats_before = {}
        self.stats_after = {}
        self.init_balance()
    
    def init_balance(self):
        print("checking dataset type for balancing...")
        if self.dataset_type in ['test', 'val']:
            global AUGMENTATION_COUNTS
            global CLASS_THRESHOLDS

            AUGMENTATION_COUNTS = {
                'low': 6,
                'medium': 4,
                'high': 2
            }

            CLASS_THRESHOLDS = {
                'low': 20,
                'medium': 50
            }
    
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
    
    def analyze_folder(self, folder_path: str=None) -> Tuple[Counter, Dict[str, str]]:
        """Analyse la répartition des classes dans un dossier donné"""
        folder = self.input_dir / 'images' if folder_path is None else Path(folder_path)
        filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_to_class = self.auto_detect_patterns(filenames)
        counts = Counter(file_to_class.values())
        total_files = len(filenames)
        total_objects = sum(counts.values())

        print(f"\n📊 RÉSULTATS ({total_files} fichiers traités)")
        print("=" * 50)

        for cls, count in counts.most_common():
            pct = (count / total_objects) * 100
            print(f"{cls:<20} : {count:7d} objets ({pct:5.1f}%)")

        print("-" * 50)
        print(f"TOTAL{' ' * 16}: {total_objects:7d} objets")
        print(f"Fichiers{' ' * 12}: {total_files} fichiers\n")

        return counts, file_to_class

    def classify_class_level(self, count: int) -> str:
        """Détermine le niveau d'une classe selon son nombre d'images"""
        if count <= CLASS_THRESHOLDS['low']:
            return 'low'
        elif count <= CLASS_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'high'
    
    def augment(self):
        """Lance l'augmentation avec multiprocessing"""
        images_dir = self.input_dir / 'images'
        labels_dir = self.input_dir / 'Annotations'
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        # Détection des classes
        filenames = [f.name for f in image_files]
        file_to_class = self.auto_detect_patterns(filenames)
        
        # Comptage des classes
        class_counter = Counter(file_to_class.values())
        self.stats_before = dict(class_counter)
        
        # Classification par niveau
        class_levels = {cls: self.classify_class_level(count) 
                       for cls, count in class_counter.items()}
        
        print(f"\n{'='*60}")
        print(f"🚀 AUGMENTATION PASCAL VOC XML")
        print(f"{'='*60}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Balance classes: {self.balance_classes}")
        print(f"Images trouvées: {len(image_files)}")
        print(f"Classes détectées: {len(class_counter)}")
        print(f"Transformations disponibles: {len(self.transforms)}")
        print(f"Workers: {self.num_workers}")
        self.analyze_folder()
        
        if self.balance_classes:
            print(f"\n📊 Distribution des classes:")
            for level in ['low', 'medium', 'high']:
                classes = [c for c, l in class_levels.items() if l == level]
                if classes:
                    print(f"  🔴 {level.upper()}: {len(classes)} classes → {AUGMENTATION_COUNTS[level]} augmentations/image")
        
        print(f"\n{'='*60}\n")
        
        # Préparer les arguments pour le multiprocessing
        args_list = [
            (img_path, labels_dir, self.output_images, self.output_annots,
             self.transforms, file_to_class, class_levels,
             self.balance_classes, self.dataset_type, AUGMENTATION_COUNTS)
            for img_path in image_files
        ]
        
        # Traitement avec multiprocessing
        print(f"🚀 Traitement en cours avec {self.num_workers} workers...\n")
        
        augmented_counter = Counter()
        
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image_worker, args_list),
                total=len(args_list),
                desc="Augmentation"
            ))
        
        # Agréger les résultats
        for counter in results:
            augmented_counter.update(counter)
        
        self.stats_after = dict(augmented_counter)
        
        total_images = len(list(self.output_images.glob('*')))
        print(f"\n✅ Augmentation terminée!")
        print(f"   Images originales: {len(image_files)}")
        print(f"   Images totales: {total_images}")
        print(f"   Ratio: x{total_images / len(image_files):.2f}")
    
    def export_statistics(self):
        """Exporte les statistiques en CSV et génère des graphiques"""
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


def process_all_splits(root_dir: str, num_workers: int = 4):
    """
    Traite automatiquement tous les splits (train, val, test)
    """
    root_path = Path(root_dir)
    splits = ['train', 'val', 'test']
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAITEMENT DE TOUS LES SPLITS")
    print(f"{'='*60}")
    print(f"Root directory: {root_dir}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    for split in splits:
        split_input = root_path / split
        
        # Vérifier si le dossier existe
        if not split_input.exists() or not (split_input / 'images').exists():
            print(f"⏭️  {split.upper()} non trouvé, passage au suivant\n")
            continue
        
        # Créer dossier de sortie
        split_output = root_path / f"augmented/{split}"
        
        print(f"\n{'='*60}")
        print(f"📁 TRAITEMENT DE {split.upper()}")
        print(f"{'='*60}\n")
        
        # Créer augmenteur
        augmenter = PascalVOCAugmenter(
            input_dir=str(split_input),
            output_dir=str(split_output),
            dataset_type=split,
            balance_classes=True,
            num_workers=num_workers
        )
        
        # Lancer augmentation
        augmenter.augment()
        
        # Exporter statistiques
        df = augmenter.export_statistics()
        
        # Générer graphiques
        plot_statistics(df, split_output)
        
        print(f"\n✅ {split.upper()} terminé!\n")
    
    print(f"\n{'='*60}")
    print("🎉 TOUS LES SPLITS TERMINÉS!")
    print(f"{'='*60}\n")


# ============= UTILISATION =============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Augmentation Pascal VOC XML avec multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Traiter tous les splits automatiquement
  python augment_pascal_voc.py -r ./dataset -w 4
  
  # Traiter un seul split
  python augment_pascal_voc.py -i ./dataset/train -o ./output -d train -w 4
        """
    )
    
    parser.add_argument(
        '-r', '--root',
        type=str,
        help='Dossier racine contenant train/, val/, test/ (traite tous les splits automatiquement)'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Dossier d\'entrée pour un split spécifique'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Dossier de sortie (seulement avec --input)'
    )
    
    parser.add_argument(
        '-d', '--dataset_type', 
        type=str,
        choices=['train', 'val', 'test'],
        default='train',
        help='Type de dataset (seulement avec --input)'
    )
    
    parser.add_argument(
        '-b', '--balance', 
        dest='balance',
        type=lambda x: str(x).lower() == 'true',
        default=True,
        help='Activer l\'équilibrage des classes (True/False)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Nombre de workers CPU pour le multiprocessing (défaut: 4)'
    )
    
    args = parser.parse_args()
    
    # Mode automatique (tous les splits)
    if args.root:
        process_all_splits(args.root, num_workers=args.workers)
    
    # Mode manuel (un seul split)
    elif args.input:
        if not args.output:
            args.output = str(Path(args.input).parent / f"{Path(args.input).name}_augmented")
        
        print("\n" + "="*60)
        print("🔧 CONFIGURATION")
        print("="*60)
        print(f"Input dir:      {args.input}")
        print(f"Output dir:     {args.output}")
        print(f"Dataset type:   {args.dataset_type}")
        print(f"Balance:        {args.balance}")
        print(f"Workers:        {args.workers}")
        print("="*60 + "\n")
        
        augmenter = PascalVOCAugmenter(
            input_dir=args.input,
            output_dir=args.output,
            dataset_type=args.dataset_type,
            balance_classes=args.balance,
            num_workers=args.workers
        )
        
        augmenter.augment()
        df = augmenter.export_statistics()
        plot_statistics(df, Path(args.output))
        
        print("\n🎉 Pipeline terminé avec succès!")
    
    else:
        parser.print_help()
        print("\n❌ Erreur: spécifiez --root OU --input")