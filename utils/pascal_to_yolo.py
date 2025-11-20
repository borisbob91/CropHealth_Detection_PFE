import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import argparse

# ============= CONVERTISSEUR VOC → YOLO =============

class VOCtoYOLOConverter:
    """Convertit Pascal VOC XML vers YOLO TXT avec métadonnées"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Créer structure de sortie
        self.output_images = self.output_dir / 'images'
        self.output_labels = self.output_dir / 'labels'
        self.output_images.mkdir(parents=True, exist_ok=True)
        self.output_labels.mkdir(parents=True, exist_ok=True)
        
        # Dictionnaire des classes
        self.class_names = []
        self.class_to_id = {}
    
    def extract_classes_from_xml(self):
        """Extrait toutes les classes uniques depuis les fichiers XML"""
        labels_dir = self.input_dir / 'Annotations'
        xml_files = list(labels_dir.glob('*.xml'))
        
        unique_classes = set()
        
        print("\n🔍 Extraction des classes depuis les XML...")
        for xml_path in tqdm(xml_files, desc="Scan XML"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    unique_classes.add(class_name)
            except Exception as e:
                print(f"⚠️ Erreur lecture {xml_path.name}: {e}")
        
        # Trier alphabétiquement
        self.class_names = sorted(list(unique_classes))
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✅ {len(self.class_names)} classes détectées")
    
    def parse_voc_xml(self, xml_path: Path):
        """Parse un fichier Pascal VOC et retourne les bboxes YOLO normalisées"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Dimensions de l'image
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            if class_name not in self.class_to_id:
                continue
            
            class_id = self.class_to_id[class_name]
            
            # Bounding box Pascal VOC [xmin, ymin, xmax, ymax]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Conversion YOLO [x_center, y_center, width, height] (normalisé 0-1)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Clip values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def convert(self):
        """Lance la conversion VOC → YOLO"""
        images_dir = self.input_dir / 'images'
        labels_dir = self.input_dir / 'labels'
        
        # Extraire les classes
        self.extract_classes_from_xml()
        
        # Lister images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"\n{'='*60}")
        print(f"🚀 CONVERSION PASCAL VOC → YOLO")
        print(f"{'='*60}")
        print(f"Images trouvées: {len(image_files)}")
        print(f"Classes: {len(self.class_names)}")
        print(f"{'='*60}\n")
        
        converted_count = 0
        skipped_count = 0
        
        for img_path in tqdm(image_files, desc="Conversion"):
            xml_path = labels_dir / (img_path.stem + '.xml')
            
            if not xml_path.exists():
                skipped_count += 1
                continue
            
            try:
                # Convertir annotations
                yolo_annotations = self.parse_voc_xml(xml_path)
                
                if len(yolo_annotations) == 0:
                    skipped_count += 1
                    continue
                
                # Copier image
                import shutil
                shutil.copy(str(img_path), str(self.output_images / img_path.name))
                
                # Sauvegarder annotations YOLO
                txt_path = self.output_labels / (img_path.stem + '.txt')
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
            
            except Exception as e:
                print(f"\n❌ Erreur sur {img_path.name}: {e}")
                skipped_count += 1
        
        print(f"\n✅ Conversion terminée!")
        print(f"   Images converties: {converted_count}")
        print(f"   Images ignorées: {skipped_count}")
        
        # Générer fichiers métadonnées
        self.save_metadata()
    
    def save_metadata(self):
        """Génère classes.txt et notes.json"""
        
        # 1. Sauvegarder classes.txt
        classes_path = self.output_dir / 'classes.txt'
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.class_names))
        print(f"\n📄 Fichier classes.txt créé: {classes_path}")
        
        # 2. Sauvegarder notes.json (format Label Studio)
        categories = [
            {"id": idx, "name": name}
            for idx, name in enumerate(self.class_names)
        ]
        
        notes_data = {
            "categories": categories,
            "info": {
                "year": 2025,
                "version": "1.0",
                "contributor": "VOC to YOLO Converter"
            }
        }
        
        notes_path = self.output_dir / 'notes.json'
        with open(notes_path, 'w', encoding='utf-8') as f:
            json.dump(notes_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Fichier notes.json créé: {notes_path}")
        
        # 3. Afficher récapitulatif
        print(f"\n{'='*60}")
        print("📊 CLASSES DÉTECTÉES")
        print(f"{'='*60}")
        for idx, name in enumerate(self.class_names):
            print(f"  {idx}: {name}")
        print(f"{'='*60}")

# ============= UTILISATION =============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Conversion Pascal VOC XML vers YOLO TXT + métadonnées',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python voc_to_yolo.py
  python voc_to_yolo.py --input ./dataset_voc --output ./dataset_yolo
  python voc_to_yolo.py -i ./data -o ./yolo_data

Structure attendue (input):
  input_dir/
  ├── images/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── labels/
      ├── image1.xml
      └── image2.xml

Structure générée (output):
  output_dir/
  ├── images/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── labels/
  │   ├── image1.txt
  │   └── image2.txt
  ├── classes.txt
  └── notes.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\train',
        help='Dossier d\'entrée contenant images/ et labels/ (Pascal VOC XML)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\train\yolo_dataset',
        help='Dossier de sortie pour le dataset YOLO'
    )
    
    args = parser.parse_args()
    
    # Afficher configuration
    print("\n" + "="*60)
    print("🔧 CONFIGURATION")
    print("="*60)
    print(f"Input dir (VOC):  {args.input}")
    print(f"Output dir (YOLO): {args.output}")
    print("="*60 + "\n")
    
    # Créer convertisseur
    converter = VOCtoYOLOConverter(
        input_dir=args.input,
        output_dir=args.output
    )
    
    # Lancer conversion
    converter.convert()
    
    print("\n🎉 Pipeline terminé avec succès!")