import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
from typing import List, Dict, Tuple
import re
import shutil

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

            generated = 0

            # =================================================================
            # 1. Toujours générer la version "resize classique"
            # =================================================================
            transform_normal = A.Compose([
                A.Resize(height=target_size[1], width=target_size[0])
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

            transformed_normal = transform_normal(image=image, bboxes=bboxes, labels=labels)

            normal_img_path = os.path.join(output_images_dir, f"{base_name}_full.jpg")
            normal_xml_path = os.path.join(output_labels_dir, f"{base_name}_full.xml")

            cv2.imwrite(normal_img_path, transformed_normal['image'])
            self._save_xml(root, transformed_normal['bboxes'], transformed_normal['labels'], target_size, normal_xml_path)
            generated += 1

            # =================================================================
            # 2. Si l'image est grande → générer aussi la version "crop zoom"
            # =================================================================
            is_large = w >= large_threshold or h >= large_threshold

            if is_large and bboxes:  # seulement si y a des objets
                # Calcul du bounding box global des objets
                all_x = [b[0] for b in bboxes] + [b[2] for b in bboxes]
                all_y = [b[1] for b in bboxes] + [b[3] for b in bboxes]
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

                transformed_crop = crop_transform(image=image, bboxes=bboxes, labels=labels)

                if len(transformed_crop['bboxes']) == len(bboxes):  # toutes les bbox visibles
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

        # Ajouter nouveaux objects
        for label, bbox in zip(labels, bboxes):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

        ET.ElementTree(root).write(output_xml_path)

 
# ======================== UTILISATION ========================
if __name__ == "__main__":
    resizer = DoubleSmartResizer()

    root_dir = r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\dataset_pascal_voc\test'
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'Annotations')

    output_root = os.path.join(root_dir, 'resized_double_dataset')
    output_images_dir = os.path.join(output_root, 'images')
    output_labels_dir = os.path.join(output_root, 'Annotations')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    total_input = 0
    total_output = 0

    print("Début du double traitement (full + zoom sur grandes images)...\n")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            total_input += 1
            image_path = os.path.join(images_dir, filename)
            xml_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.xml')

            if not os.path.exists(xml_path):
                print(f"XML manquant: {filename}")
                continue

            generated = resizer.process_single_image(image_path, xml_path, output_images_dir, output_labels_dir)
            total_output += generated

            status = "1 image" if generated == 1 else "2 images (full + zoom)"
            print(f"[{total_input}] {filename} → {status}")

    print("\n" + "="*60)
    print(f"FINI !")
    print(f"Images en entrée  : {total_input}")
    print(f"Images en sortie  : {total_output}  🎉")
    print(f"→ +{total_output - total_input} images bonus grâce au zoom intelligent")
    print(f"Dossier : {output_root}")
    print("="*60)