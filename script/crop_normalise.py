import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
from typing import List, Dict, Tuple
import re
import shutil

from utils.fix_xml_path import corriger_filename_dans_xml

class DoubleSmartResizer:
    def __init__(self):
        self.pattern_regex = re.compile(r'(?:^|-)([A-Z]+_[A-Z]+_[A-Z]+)(?:_\d+)?')

    def auto_detect_patterns(self, filenames: List[str]) -> Dict[str, str]:
        file_to_class = {}
        for filename in filenames:
            basename = os.path.splitext(filename)[0]
            match = self.pattern_regex.search(basename)
            if match:
                file_to_class[filename] = match.group(1)
            else:
                file_to_class[filename] = 'unknown'
        return file_to_class

    def validate_bboxes(self, bboxes: List[List[float]], image_size: Tuple[int, int], stage: str) -> bool:
        """
        Valide que toutes les bounding boxes sont normalisées et dans les limites
        """
        width, height = image_size
        valid = True
        
        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            
            # Vérifier les types
            if not all(isinstance(coord, (int, float)) for coord in [xmin, ymin, xmax, ymax]):
                print(f"  ⚠️ {stage}: Bbox {i} contient des types invalides: {bbox}")
                valid = False
            
            # Vérifier l'ordre des coordonnées
            if xmin >= xmax or ymin >= ymax:
                print(f"  ⚠️ {stage}: Bbox {i} a des coordonnées inversées: {bbox}")
                valid = False
            
            # Vérifier les limites de l'image
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                print(f"  ⚠️ {stage}: Bbox {i} dépasse l'image {image_size}: {bbox}")
                valid = False
            
            # Vérifier la taille minimale
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            if bbox_width < 2 or bbox_height < 2:
                print(f"  ⚠️ {stage}: Bbox {i} trop petite: {bbox_width}x{bbox_height}")
                valid = False
                
            # Vérifier la surface minimale
            area = bbox_width * bbox_height
            if area < 4:  # au moins 4 pixels
                print(f"  ⚠️ {stage}: Bbox {i} surface trop petite: {area}px²")
                valid = False
        
        if valid:
            print(f"  ✓ {stage}: {len(bboxes)} bboxes validées")
        else:
            print(f"  ❌ {stage}: Problèmes détectés dans les bboxes")
            
        return valid

    def normalize_bboxes(self, bboxes: List[List[float]], image_size: Tuple[int, int]) -> List[List[float]]:
        """
        Normalise les bounding boxes pour s'assurer qu'elles sont dans les limites
        """
        width, height = image_size
        normalized_bboxes = []
        
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            
            # Clamper les coordonnées dans les limites de l'image
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(1, min(xmax, width))
            ymax = max(1, min(ymax, height))
            
            # S'assurer que xmin < xmax et ymin < ymax
            if xmin >= xmax:
                xmin, xmax = max(0, xmax - 1), min(width, xmin + 1)
            if ymin >= ymax:
                ymin, ymax = max(0, ymax - 1), min(height, ymin + 1)
            
            # S'assurer que la bbox a une taille minimale
            if xmax - xmin < 2:
                xmax = min(width, xmin + 2)
            if ymax - ymin < 2:
                ymax = min(height, ymin + 2)
            
            normalized_bboxes.append([xmin, ymin, xmax, ymax])
        
        return normalized_bboxes

    def process_single_image(
        self,
        image_path: str,
        xml_path: str,
        output_images_dir: str,
        output_labels_dir: str,
        target_size: Tuple[int, int] = (800, 800),
        large_threshold: int = 1000,
    ) -> int:
        """
        Traite une image et retourne le nombre d'images générées (1 ou 2)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erreur lecture {image_path}")
                return 0

            h, w = image.shape[:2]
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Lecture XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            bboxes = []
            labels = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)

            # Normaliser les bboxes originales
            original_bboxes = self.normalize_bboxes(bboxes, (w, h))
            
            # Valider les bboxes originales
            if not self.validate_bboxes(original_bboxes, (w, h), "Original"):
                print(f"  ❌ Bboxes originales invalides pour {base_name}")
                return 0

            generated = 0

            # =================================================================
            # 1. Toujours générer la version "resize classique"
            # =================================================================
            transform_normal = A.Compose([
                A.Resize(height=target_size[1], width=target_size[0])
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

            transformed_normal = transform_normal(image=image, bboxes=original_bboxes, labels=labels)

            # Valider les bboxes après resize
            if not self.validate_bboxes(transformed_normal['bboxes'], target_size, "After Resize"):
                print(f"  ❌ Bboxes resize invalides pour {base_name}")
                return 0

            normal_img_path = os.path.join(output_images_dir, f"{base_name}_full.jpg")
            normal_xml_path = os.path.join(output_labels_dir, f"{base_name}_full.xml")

            cv2.imwrite(normal_img_path, transformed_normal['image'])
            self._save_xml(root, transformed_normal['bboxes'], transformed_normal['labels'], target_size, normal_xml_path)
            generated += 1

            # =================================================================
            # 2. Si l'image est grande → générer aussi la version "crop zoom"
            # =================================================================
            is_large = w >= large_threshold or h >= large_threshold

            if is_large and original_bboxes:  # seulement si y a des objets
                # Calcul du bounding box global des objets
                all_x = [b[0] for b in original_bboxes] + [b[2] for b in original_bboxes]
                all_y = [b[1] for b in original_bboxes] + [b[3] for b in original_bboxes]
                global_xmin, global_xmax = min(all_x), max(all_x)
                global_ymin, global_ymax = min(all_y), max(all_y)

                obj_w = global_xmax - global_xmin
                obj_h = global_ymax - global_ymin

                margin = 1.4  # 40% de marge autour des objets
                crop_w = max(obj_w * margin, 500)
                crop_h = max(obj_h * margin, 500)

                # Forcer ratio carré
                if crop_w > crop_h:
                    crop_h = crop_w
                else:
                    crop_w = crop_h

                center_x = (global_xmin + global_xmax) / 2
                center_y = (global_ymin + global_ymax) / 2

                x1 = int(center_x - crop_w / 2)
                y1 = int(center_y - crop_h / 2)
                x2 = int(center_x + crop_w / 2)
                y2 = int(center_y + crop_h / 2)

                # Clamp + sécurité pour ne perdre aucune bbox
                x1 = max(0, min(x1, global_xmin - 50))
                y1 = max(0, min(y1, global_ymin - 50))
                x2 = min(w, max(x2, global_xmax + 50))
                y2 = min(h, max(y2, global_ymax + 50))

                crop_transform = A.Compose([
                    A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                    A.Resize(height=target_size[1], width=target_size[0])
                ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.9, label_fields=['labels']))

                transformed_crop = crop_transform(image=image, bboxes=original_bboxes, labels=labels)

                # Valider les bboxes après crop
                if not self.validate_bboxes(transformed_crop['bboxes'], target_size, "After Crop"):
                    print(f"  ❌ Bboxes crop invalides pour {base_name}")
                    return generated  # On retourne seulement les images déjà générées

                if len(transformed_crop['bboxes']) == len(original_bboxes):  # toutes les bbox visibles
                    crop_img_path = os.path.join(output_images_dir, f"{base_name}_zoom.jpg")
                    crop_xml_path = os.path.join(output_labels_dir, f"{base_name}_zoom.xml")

                    cv2.imwrite(crop_img_path, transformed_crop['image'])
                    self._save_xml(root, transformed_crop['bboxes'], transformed_crop['labels'], target_size, crop_xml_path)
                    generated += 1
                else:
                    print(f"   → Crop a perdu des objets sur {base_name}, version zoom annulée")

            return generated

        except Exception as e:
            print(f"Erreur critique {image_path}: {e}")
            return 0

    def _save_xml(self, root, bboxes, labels, target_size, output_xml_path):
        # Nettoyer anciens objects
        for obj in root.findall("object"):
            root.remove(obj)

        # Mettre à jour taille
        root.find("size").find("width").text = str(target_size[0])
        root.find("size").find("height").text = str(target_size[1])

        # Ajouter nouveaux objects avec bboxes normalisées
        for label, bbox in zip(labels, bboxes):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            bndbox = ET.SubElement(obj, "bndbox")
            
            # S'assurer que les coordonnées sont entières et dans les limites
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, min(int(xmin), target_size[0] - 1))
            ymin = max(0, min(int(ymin), target_size[1] - 1))
            xmax = max(1, min(int(xmax), target_size[0]))
            ymax = max(1, min(int(ymax), target_size[1]))
            
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        ET.ElementTree(root).write(output_xml_path)


# ======================== UTILISATION ========================
if __name__ == "__main__":
    resizer = DoubleSmartResizer()
    base_dir = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc_ac"
    root_dir = os.path.join(base_dir, 'test')
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'Annotations')

    output_root = os.path.join(root_dir, 'resized_double_dataset')
    output_images_dir = os.path.join(output_root, 'images')
    output_labels_dir = os.path.join(output_root, 'Annotations')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    total_input = 0
    total_output = 0
    validation_errors = 0

    print("Début du double traitement avec validation des bboxes...\n")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            total_input += 1
            image_path = os.path.join(images_dir, filename)
            xml_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.xml')

            if not os.path.exists(xml_path):
                print(f"XML manquant: {filename}")
                continue

            print(f"\n[{total_input}] Traitement de {filename}")
            generated = resizer.process_single_image(image_path, xml_path, output_images_dir, output_labels_dir)
            
            if generated == 0:
                validation_errors += 1
                print(f"  ❌ Échec de validation pour {filename}")
            else:
                total_output += generated
                status = "1 image" if generated == 1 else "2 images (full + zoom)"
                print(f"  ✅ {filename} → {status}")
    corriger_filename_dans_xml()
    print("\n" + "="*60)
    print(f"FINI !")
    print(f"Images en entrée  : {total_input}")
    print(f"Images en sortie  : {total_output}  🎉")
    print(f"Erreurs de validation : {validation_errors}")
    print(f"→ +{total_output - total_input} images bonus grâce au zoom intelligent")
    print(f"Dossier : {output_root}")
    print("="*60)