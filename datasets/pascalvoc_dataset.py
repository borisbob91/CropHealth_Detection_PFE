"""
CropHealth Detection - Pascal VOC Dataset (CORRIGÉ)
Gère correctement les images sans annotations après augmentation
"""
import os
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from configs.model_configs import NUM_CLASSES, CLASS_NAMES

class PascalVOCDataset(Dataset):
    """
    Dataset Pascal VOC format (XML) → PyTorch
    Applique Albumentations transforms avec gestion robuste des cas limites
    
    Structure attendue:
        data_root/
        ├── images/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── Annotations/
            ├── img1.xml
            └── img2.xml
    """
    def __init__(self, img_root, ann_root, class_names, transforms=None, min_box_size=1.0):
        """
        Args:
            img_root: chemin vers images/
            ann_root: chemin vers Annotations/
            class_names: liste des noms de classes (ex: ['aphid', 'bollworm', ...])
            transforms: pipeline Albumentations (get_albu_transform)
            min_box_size: taille minimale des boxes (en pixels) pour être valides
        """
        self.img_root = Path(img_root)
        self.ann_root = Path(ann_root)
        self.class_names = class_names
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # background = 0
        self.transforms = transforms
        self.min_box_size = min_box_size
        
        # Lister annotations XML
        self.annotations = sorted(list(self.ann_root.glob('*.xml')))
        
        # Statistiques pour debugging
        self.empty_images_count = 0
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Charger annotation XML
        xml_path = self.annotations[idx]
        annotation = self._parse_xml(xml_path)
        
        # Charger image
        img_name = annotation['filename']
        img_path = self.img_root / img_name
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        
        # Extraire boxes et labels
        boxes = []
        labels = []
        
        for obj in annotation['objects']:
            class_name = obj['name']
            
            # Ignorer classes inconnues
            if class_name not in self.class_to_idx:
                continue
            
            bbox = obj['bbox']
            # Vérifier que la box est valide
            if self._is_valid_box(bbox, w, h):
                boxes.append(bbox)
                labels.append(self.class_to_idx[class_name])
        
        # Cas 1: Image sans annotations valides AVANT transformation
        if len(boxes) == 0:
            self.empty_images_count += 1
            # Retourner une image avec une box factice (sera filtrée par le collate_fn)
            boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            labels = [0]  # background
        else:
            boxes = np.array(boxes, dtype=np.float32)
        
        # Appliquer Albumentations
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Cas 2: Toutes les boxes ont été supprimées par l'augmentation
        if len(boxes) == 0:
            self.empty_images_count += 1
            # Créer une box factice centrée
            img_h, img_w = img.shape[1:3] if isinstance(img, torch.Tensor) else img.shape[:2]
            boxes = [[float(img_w) * 0.4, float(img_h) * 0.4, 
                     float(img_w) * 0.6, float(img_h) * 0.6]]
            labels = [0]  # background
        
        # Conversion en tenseurs PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Validation finale
        assert boxes.shape[0] == labels.shape[0], f"Mismatch: {boxes.shape[0]} boxes vs {labels.shape[0]} labels"
        assert boxes.shape[1] == 4, f"Invalid box shape: {boxes.shape}"
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        return img, target
    
    def _is_valid_box(self, bbox, img_w, img_h):
        """Vérifie qu'une box est valide"""
        xmin, ymin, xmax, ymax = bbox
        
        # Vérifier ordre
        if xmax <= xmin or ymax <= ymin:
            return False
        
        # Vérifier taille minimale
        if (xmax - xmin) < self.min_box_size or (ymax - ymin) < self.min_box_size:
            return False
        
        # Vérifier limites image
        if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
            return False
        
        return True
    
    def _parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Image info
        filename = root.find('filename').text
        
        # Objects
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return {
            'filename': filename,
            'objects': objects
        }
    
    def get_class_names(self):
        """Retourne les noms de classes"""
        return self.class_names
    
    def print_stats(self):
        """Affiche les statistiques du dataset"""
        print(f"📊 Dataset stats:")
        print(f"   Total images: {len(self)}")
        print(f"   Images sans boxes valides: {self.empty_images_count}")
        if self.empty_images_count > 0:
            print(f"   ⚠️ {self.empty_images_count}/{len(self)} images ont des boxes factices")


def collate_fn_filter_empty(batch):
    """
    Collate function qui FILTRE les images sans annotations valides
    Utiliser avec: DataLoader(..., collate_fn=collate_fn_filter_empty)
    """
    # Filtrer les samples avec seulement des boxes background (label 0)
    filtered_batch = []
    for img, target in batch:
        # Garder seulement si au moins une box non-background
        if len(target['labels']) > 0 and (target['labels'] > 0).any():
            filtered_batch.append((img, target))
    
    # Si tout a été filtré, garder au moins un sample pour éviter les erreurs
    if len(filtered_batch) == 0:
        filtered_batch = [batch[0]]
    
    # Séparer images et targets
    images = [item[0] for item in filtered_batch]
    targets = [item[1] for item in filtered_batch]
    
    return images, targets


def collate_fn_standard(batch):
    """
    Collate function standard pour PyTorch detection models
    Garde toutes les images, même celles avec boxes factices
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# Exemple d'utilisation
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Classes exemple
    class_names = ['aphid', 'armyworm', 'beetle', 'bollworm', 'grasshopper']
    path = Path('C:\\Users\\BorisBob\\Desktop\\detection\\dataset_split\\label_studio\\pascal_voc_orignal')
    # Dataset
    dataset = PascalVOCDataset(
        img_root=os.path.join(path, 'train/images'),
        ann_root=os.path.join(path, 'train/Annotations'),
        class_names=CLASS_NAMES,
        transforms=None  # Ajouter vos transforms Albumentations ici
    )
    
    # DataLoader avec filtrage
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_filter_empty  # ← Filtre les images vides
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test
    for images, targets in train_loader:
        print(f"Batch: {len(images)} images")
        for i, target in enumerate(targets):
            print(f"  Image {i}: {len(target['boxes'])} boxes, labels: {target['labels'].tolist()}")
        break
    
    # Afficher stats
    dataset.print_stats()