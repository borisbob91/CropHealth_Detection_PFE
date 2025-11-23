import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
from typing import Tuple, List, Optional
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
from functools import partial
import shutil

class UltimateSmartResizer:
    def __init__(self, large_threshold: int = 1000):
        self.large_threshold = large_threshold
        self.target_size = (800, 800)

    def _get_centroids(self, bboxes):
        """Calcule les centroïdes des bounding boxes"""
        return np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes])

    def _detect_clusters(self, bboxes, image_shape, min_ratio=0.25):
        """Détecte si les objets sont en clusters séparés"""
        if len(bboxes) < 2:
            return None
        centroids = self._get_centroids(bboxes)
        h, w = image_shape[:2]
        centroids_norm = centroids / np.array([w, h])
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(centroids_norm)
        if np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) > min_ratio:
            return np.where(kmeans.labels_ == 0)[0], np.where(kmeans.labels_ == 1)[0]
        return None

    def _crop_and_resize(self, image, bboxes, labels, margin=1.6):
        """Crop et resize avec normalisation des coordonnées"""
        if not bboxes:
            return None, None, None
        
        xs = [x for b in bboxes for x in (b[0], b[2])]
        ys = [y for b in bboxes for y in (b[1], b[3])]
        center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w_obj, h_obj = max(xs) - min(xs), max(ys) - min(ys)
        crop_w, crop_h = max(w_obj * margin, 600), max(h_obj * margin, 600)
        h, w = image.shape[:2]
        
        x1, y1 = max(0, int(center_x - crop_w / 2)), max(0, int(center_y - crop_h / 2))
        x2, y2 = min(w, int(center_x + crop_w / 2)), min(h, int(center_y + crop_h / 2))

        transform = A.Compose([
            A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
            A.Resize(height=800, width=800)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))

        try:
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
            if len(transformed['bboxes']) > 0:
                return transformed['image'], transformed['bboxes'], transformed['labels']
        except Exception as e:
            print(f"    ⚠️  Erreur transformation: {e}")
        
        return None, None, None

    def _resize_full_image(self, image, bboxes, labels):
        """Resize l'image complète avec normalisation correcte des bboxes"""
        h, w = image.shape[:2]
        
        transform = A.Compose([
            A.Resize(height=800, width=800)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))
        
        try:
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
            return transformed['image'], transformed['bboxes'], transformed['labels']
        except Exception as e:
            print(f"    ⚠️  Erreur resize full: {e}")
            return None, None, None

    def _create_xml(self, filename: str, bboxes: List, labels: List, img_shape: Tuple) -> ET.Element:
        """Crée un XML complet avec toutes les informations correctes"""
        root = ET.Element("annotation")
        
        # Folder
        ET.SubElement(root, "folder").text = "images"
        
        # Filename
        ET.SubElement(root, "filename").text = filename
        
        # Path
        ET.SubElement(root, "path").text = filename
        
        # Source
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        
        # Size (toujours 800x800 après resize)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "800"
        ET.SubElement(size, "height").text = "800"
        ET.SubElement(size, "depth").text = "3"
        
        # Segmented
        ET.SubElement(root, "segmented").text = "0"
        
        # Objects avec coordonnées normalisées
        for label, bbox in zip(labels, bboxes):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(label)
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bnd = ET.SubElement(obj, "bndbox")
            # S'assurer que les coordonnées sont dans l'image
            xmin = max(1, min(799, int(round(bbox[0]))))
            ymin = max(1, min(799, int(round(bbox[1]))))
            xmax = max(1, min(800, int(round(bbox[2]))))
            ymax = max(1, min(800, int(round(bbox[3]))))
            
            # Vérifier que xmax > xmin et ymax > ymin
            if xmax <= xmin:
                xmax = xmin + 1
            if ymax <= ymin:
                ymax = ymin + 1
            
            ET.SubElement(bnd, "xmin").text = str(xmin)
            ET.SubElement(bnd, "ymin").text = str(ymin)
            ET.SubElement(bnd, "xmax").text = str(xmax)
            ET.SubElement(bnd, "ymax").text = str(ymax)
        
        return root

    def _save_xml(self, root: ET.Element, xml_path: str):
        """Sauvegarde le XML avec indentation"""
        self._indent_xml(root)
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

    def _indent_xml(self, elem, level=0):
        """Indente le XML pour le rendre lisible"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def process_image(self, img_path, xml_path, out_img_dir, out_xml_dir, base_name):
        """Traite une image et génère les versions augmentées"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ❌ Impossible de lire: {img_path}")
                return 0
            
            h, w = image.shape[:2]
            
            # Parse XML
            tree = ET.parse(xml_path)
            root_template = tree.getroot()

            bboxes, labels = [], []
            for obj in root_template.findall("object"):
                name = obj.find("name")
                if name is None:
                    continue
                b = obj.find("bndbox")
                if b is None:
                    continue
                
                try:
                    xmin = float(b.find("xmin").text)
                    ymin = float(b.find("ymin").text)
                    xmax = float(b.find("xmax").text)
                    ymax = float(b.find("ymax").text)
                    
                    # Validation basique
                    if xmax > xmin and ymax > ymin:
                        bboxes.append([xmin, ymin, xmax, ymax])
                        labels.append(name.text)
                except (ValueError, AttributeError) as e:
                    print(f"    ⚠️  Bbox invalide ignorée: {e}")
                    continue

            if not bboxes:
                print(f"  ⚠️  Aucune bbox valide dans {base_name}")
                return 0

            generated = 0
            is_large = max(w, h) >= self.large_threshold

            # 1. Version FULL - Resize complet avec normalisation
            full_img, full_bb, full_lb = self._resize_full_image(image, bboxes, labels)
            if full_img is not None and full_bb:
                full_filename = f"{base_name}_full.jpg"
                cv2.imwrite(os.path.join(out_img_dir, full_filename), full_img)
                
                xml_root = self._create_xml(full_filename, full_bb, full_lb, full_img.shape)
                self._save_xml(xml_root, os.path.join(out_xml_dir, f"{base_name}_full.xml"))
                generated += 1

            if not is_large:
                return generated

            # 2. Version ZOOM - Crop global avec normalisation
            z_img, z_bb, z_lb = self._crop_and_resize(image, bboxes, labels, margin=1.5)
            if z_img is not None and z_bb:
                zoom_filename = f"{base_name}_zoom.jpg"
                cv2.imwrite(os.path.join(out_img_dir, zoom_filename), z_img)
                
                xml_root = self._create_xml(zoom_filename, z_bb, z_lb, z_img.shape)
                self._save_xml(xml_root, os.path.join(out_xml_dir, f"{base_name}_zoom.xml"))
                generated += 1

            # 3. Versions SPLIT si clusters détectés
            clusters = self._detect_clusters(bboxes, image.shape)
            if clusters:
                idx1, idx2 = clusters
                for idx, suffix in zip([idx1, idx2], ["top", "bottom"]):
                    g_bboxes = [bboxes[i] for i in idx]
                    g_labels = [labels[i] for i in idx]
                    c_img, c_bb, c_lb = self._crop_and_resize(image, g_bboxes, g_labels, margin=1.8)
                    if c_img is not None and c_bb:
                        split_filename = f"{base_name}_{suffix}.jpg"
                        cv2.imwrite(os.path.join(out_img_dir, split_filename), c_img)
                        
                        xml_root = self._create_xml(split_filename, c_bb, c_lb, c_img.shape)
                        self._save_xml(xml_root, os.path.join(out_xml_dir, f"{base_name}_{suffix}.xml"))
                        generated += 1

            return generated
            
        except Exception as e:
            print(f"  ❌ Erreur sur {base_name}: {e}")
            return 0


def process_single_image(args):
    """Fonction wrapper pour le multiprocessing"""
    img_file, src_img, src_ann, out_img, out_ann, resizer = args
    
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        return 0, img_file, 0
    
    base = os.path.splitext(img_file)[0]
    img_path = os.path.join(src_img, img_file)
    xml_path = os.path.join(src_ann, base + ".xml")
    
    if not os.path.exists(xml_path):
        return 0, img_file, 0
    
    gen = resizer.process_image(img_path, xml_path, out_img, out_ann, base)
    return 1, img_file, gen


def resize_entire_dataset(root_dataset_path: str, num_workers: int = 4):
    """
    Traite tout le dataset avec multiprocessing
    
    Args:
        root_dataset_path: Chemin racine du dataset
        num_workers: Nombre de workers CPU (par défaut 4)
    """
    resizer = UltimateSmartResizer(large_threshold=1000)

    output_root = os.path.join(root_dataset_path, "resized_ultimatex4")
    splits = ['train', 'val', 'test']

    total_in, total_out = 0, 0

    print(f"\n{'='*60}")
    print(f"🚀 DÉMARRAGE DU TRAITEMENT AVEC {num_workers} WORKERS")
    print(f"{'='*60}\n")

    for split in splits:
        src_img = os.path.join(root_dataset_path, split, "images")
        src_ann = os.path.join(root_dataset_path, split, "Annotations")
        
        if not (os.path.exists(src_img) and os.path.exists(src_ann)):
            print(f"⏭️  {split.upper()} non trouvé, passage au suivant")
            continue

        out_img = os.path.join(output_root, split, "images")
        out_ann = os.path.join(output_root, split, "Annotations")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_ann, exist_ok=True)

        img_files = [f for f in os.listdir(src_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 Traitement de {split.upper()}")
        print(f"   Images à traiter: {len(img_files)}")
        print(f"   Workers: {num_workers}")
        print("-" * 60)

        # Préparer les arguments pour le multiprocessing
        args_list = [(img_file, src_img, src_ann, out_img, out_ann, resizer) 
                     for img_file in img_files]

        # Traitement en parallèle
        split_in, split_out = 0, 0
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_image, args_list)
        
        # Collecter les résultats
        for processed, img_file, gen in results:
            split_in += processed
            split_out += gen
            if gen > 0:
                print(f"  ✅ {img_file} → {gen} versions")
            elif processed:
                print(f"  ⚠️  {img_file} → 0 versions (problème)")

        total_in += split_in
        total_out += split_out
        
        print(f"\n📊 Résumé {split.upper()}: {split_in} images → {split_out} versions (+{split_out - split_in} bonus)")

    print("\n" + "="*60)
    print("🎉 TOUT EST FINI BORIS !")
    print(f"📥 Images d'origine : {total_in}")
    print(f"📤 Images générées  : {total_out}")
    print(f"➕ Images bonus     : +{total_out - total_in}")
    print(f"📂 Dossier sortie   : {output_root}")
    print("="*60 + "\n")


# ======================== LANCE ÇA ========================
if __name__ == "__main__":
    # METS JUSTE TON DOSSIER RACINE ICI
    root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc"
    
    # Lance avec 4 workers (ou change le nombre si tu veux)
    resize_entire_dataset(root, num_workers=4)