import os
import re
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

class PascalVOCVisualizer:
    def __init__(self, root_dir: str):
        """
        Initialise le visualiseur Pascal VOC
        
        Args:
            root_dir: Chemin vers le dossier contenant 'images' et 'annotations'
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'images'
        self.annotations_dir = self.root_dir / 'annotations'
        
        if not self.images_dir.exists():
            raise ValueError(f"Le dossier images n'existe pas: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise ValueError(f"Le dossier annotations n'existe pas: {self.annotations_dir}")
    
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
                # Images sans convention → classe "unknown"
                file_to_class[filename] = 'unknown'
        
        return file_to_class
    
    def parse_xml_annotation(self, xml_path: str) -> List[Dict]:
        """
        Parse un fichier XML Pascal VOC
        
        Returns:
            Liste de dictionnaires contenant les informations des bounding boxes
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            annotations.append({
                'class': name,
                'bbox': (xmin, ymin, xmax, ymax)
            })
        
        return annotations
    
    def group_images_by_class(self) -> Dict[str, List[str]]:
        """Groupe les images par classe détectée"""
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        file_to_class = self.auto_detect_patterns(image_files)
        
        # Grouper par classe
        class_to_files = {}
        for filename, class_name in file_to_class.items():
            if class_name not in class_to_files:
                class_to_files[class_name] = []
            class_to_files[class_name].append(filename)
        
        return class_to_files
    
    def get_random_color(self, seed_string: str) -> Tuple[float, float, float]:
        """Génère une couleur aléatoire basée sur une chaîne (pour cohérence)"""
        np.random.seed(hash(seed_string) % (2**32))
        return tuple(np.random.rand(3))
    
    def visualize_class(self, class_name: str, num_batches: int = 16):
        """
        Visualise les images d'une classe avec leurs annotations
        
        Args:
            class_name: Nom de la classe à visualiser
            num_batches: Nombre de batchs à créer (défaut: 16)
        """
        class_to_files = self.group_images_by_class()
        
        if class_name not in class_to_files:
            print(f"Classe '{class_name}' non trouvée!")
            print(f"Classes disponibles: {list(class_to_files.keys())}")
            return
        
        image_files = class_to_files[class_name]
        total_images = len(image_files)
        
        # Calculer le nombre d'images par batch
        # images_per_batch = max(1, total_images // num_batches)
        images_per_batch = 20
        actual_batches = (total_images + images_per_batch - 1) // images_per_batch
        
        print(f"\nClasse: {class_name}")
        print(f"Nombre total d'images: {total_images}")
        print(f"Images par batch: {images_per_batch}")
        print(f"Nombre de batchs: {actual_batches}")
        
        # Générer un mapping couleur pour chaque classe d'objet
        all_classes = set()
        for filename in image_files:
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = self.annotations_dir / xml_filename
            if xml_path.exists():
                annotations = self.parse_xml_annotation(str(xml_path))
                for ann in annotations:
                    all_classes.add(ann['class'])
        
        # Créer un dictionnaire de couleurs aléatoires pour chaque classe
        color_map = {cls: self.get_random_color(cls) for cls in all_classes}
        
        for batch_idx in range(actual_batches):
            start_idx = batch_idx * images_per_batch
            end_idx = min(start_idx + images_per_batch, total_images)
            batch_files = image_files[start_idx:end_idx]
            
            # Calculer la disposition de la grille (4 colonnes)
            n_images = len(batch_files)
            cols = 4
            rows = (n_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
            fig.suptitle(f'Classe: {class_name} (Batch {batch_idx+1}/{actual_batches})', 
                        fontsize=16, fontweight='bold')
            
            # Aplatir axes pour itération facile
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for idx, filename in enumerate(batch_files):
                ax = axes[idx]
                
                # Charger l'image
                img_path = self.images_dir / filename
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Afficher l'image
                ax.imshow(img_rgb)
                
                # Parser les annotations
                xml_filename = os.path.splitext(filename)[0] + '.xml'
                xml_path = self.annotations_dir / xml_filename
                
                if xml_path.exists():
                    annotations = self.parse_xml_annotation(str(xml_path))
                    
                    # Dessiner les bounding boxes
                    for ann in annotations:
                        xmin, ymin, xmax, ymax = ann['bbox']
                        width = xmax - xmin
                        height = ymax - ymin
                        
                        color = color_map.get(ann['class'], (1, 1, 0))
                        
                        # Rectangle
                        rect = patches.Rectangle((xmin, ymin), width, height,
                                                linewidth=2, edgecolor=color,
                                                facecolor='none')
                        ax.add_patch(rect)
                        
                        # Label
                        ax.text(xmin, ymin-5, ann['class'],
                               color='white', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=color, alpha=0.8))
                
                # Titre avec nom de fichier
                ax.set_title(filename, fontsize=8, pad=3)
                ax.axis('off')
            
            # Cacher les axes vides
            for idx in range(len(batch_files), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def visualize_all_classes(self, num_batches: int = 16):
        """
        Visualise toutes les classes détectées
        
        Args:
            num_batches: Nombre de batchs par classe (défaut: 16)
        """
        class_to_files = self.group_images_by_class()
        
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ DES CLASSES DÉTECTÉES")
        print(f"{'='*60}")
        
        for class_name, files in class_to_files.items():
            print(f"  - {class_name}: {len(files)} images")
        
        print(f"{'='*60}\n")
        
        for class_name in class_to_files.keys():
            self.visualize_class(class_name, num_batches)
            print("\n" + "="*60 + "\n")


# ============================
# UTILISATION DU SCRIPT
# ============================

if __name__ == "__main__":
    # Chemin vers votre dossier racine contenant 'images' et 'annotations'
    root_directory = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\dataset_pascal_voc\test\resized_double_dataset\aug"  
    
    try:
        # Créer le visualiseur
        visualizer = PascalVOCVisualizer(root_directory)
        
        # Option 1: Visualiser toutes les classes (16 batchs par classe)
        visualizer.visualize_all_classes(num_batches=16)
        
        # Option 2: Visualiser une classe spécifique (16 batchs)
        # visualizer.visualize_class('IMG_R_HA', num_batches=16)
        
    except Exception as e:
        print(f"Erreur: {e}")
        print("\nAssurez-vous que:")
        print("  1. Le chemin du dossier racine est correct")
        print("  2. Les sous-dossiers 'images' et 'annotations' existent")
        print("  3. Les fichiers XML correspondent aux images")