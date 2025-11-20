import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import argparse

# ============= VISUALISATION YOLO =============

class YOLOVisualizer:
    """Visualise des images YOLO avec bounding boxes"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Charger classes
        self.class_names = self.load_classes()
        
        # Générer couleurs aléatoires pour chaque classe
        np.random.seed(42)
        self.colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) 
                      for i in range(len(self.class_names))}
    
    def load_classes(self) -> List[str]:
        """Charge les noms de classes depuis classes.txt ou notes.json"""
        classes_txt = self.dataset_dir / 'classes.txt'
        notes_json = self.dataset_dir / 'notes.json'
        
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
    
    def parse_yolo_label(self, txt_path: Path, img_width: int, img_height: int) -> List[Tuple]:
        """Parse fichier YOLO et convertit en coordonnées pixel"""
        annotations = []
        
        if not txt_path.exists():
            return annotations
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convertir en coordonnées pixel
                    x_center_px = int(x_center * img_width)
                    y_center_px = int(y_center * img_height)
                    w_px = int(width * img_width)
                    h_px = int(height * img_height)
                    
                    # Calculer coins
                    x1 = int(x_center_px - w_px / 2)
                    y1 = int(y_center_px - h_px / 2)
                    x2 = int(x_center_px + w_px / 2)
                    y2 = int(y_center_px + h_px / 2)
                    
                    annotations.append((class_id, x1, y1, x2, y2))
        
        return annotations
    
    def draw_boxes(self, image: np.ndarray, annotations: List[Tuple]) -> np.ndarray:
        """Dessine les bounding boxes sur l'image"""
        img_copy = image.copy()
        
        for class_id, x1, y1, x2, y2 in annotations:
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Dessiner rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            label = f"{class_name}"
            
            # Background pour le texte
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_copy
    
    def visualize_random(self, num_images: int = 10):
        """Visualise num_images images aléatoires avec leurs annotations"""
        # Lister toutes les images
        image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        
        if len(image_files) == 0:
            print("❌ Aucune image trouvée dans", self.images_dir)
            return
        
        # Sélectionner aléatoirement
        num_images = min(num_images, len(image_files))
        selected_files = random.sample(image_files, num_images)
        
        print(f"\n{'='*60}")
        print(f"🖼️  VISUALISATION YOLO")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_dir}")
        print(f"Images totales: {len(image_files)}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Sélection aléatoire: {num_images} images")
        print(f"{'='*60}\n")
        
        # Calculer grille
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, img_path in enumerate(selected_files):
            # Lire image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]
            
            # Lire annotations
            txt_path = self.labels_dir / (img_path.stem + '.txt')
            annotations = self.parse_yolo_label(txt_path, img_w, img_h)
            
            # Dessiner boxes
            image_with_boxes = self.draw_boxes(image, annotations)
            
            # Afficher
            axes[idx].imshow(image_with_boxes)
            axes[idx].axis('off')
            axes[idx].set_title(f"{img_path.name}\n{len(annotations)} objets", fontsize=9)
        
        # Masquer axes vides
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Visualisation terminée!")

# ============= UTILISATION =============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualisation YOLO - 10 images aléatoires avec bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python visualize_yolo.py
  python visualize_yolo.py --dataset ./yolo_dataset
  python visualize_yolo.py --dataset ./yolo_dataset --num_images 15
  python visualize_yolo.py -d ./data -n 20

Google Colab:
  !python visualize_yolo.py --dataset /content/drive/MyDrive/yolo_dataset --num_images 10

Structure attendue:
  dataset_dir/
  ├── images/
  ├── labels/
  ├── classes.txt (ou notes.json)
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\train',
        help='Chemin du dossier YOLO (contenant images/, labels/, classes.txt)'
    )
    
    parser.add_argument(
        '--num_images', '-n',
        type=int,
        default=6,
        help='Nombre d\'images aléatoires à visualiser (défaut: 10)'
    )
    
    args = parser.parse_args()
    
    # Créer visualiseur
    visualizer = YOLOVisualizer(dataset_dir=args.dataset)
    
    # Visualiser
    visualizer.visualize_random(num_images=args.num_images)
    visualizer.visualize_random(num_images=args.num_images)