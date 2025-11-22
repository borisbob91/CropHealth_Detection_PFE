# ultimate_resizer_full_dataset.py

import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans

class UltimateSmartResizer:
    def __init__(self, large_threshold: int = 1500):
        self.large_threshold = large_threshold
        self.target_size = (800, 800)

    def _get_centroids(self, bboxes):
        return np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes])

    def _detect_clusters(self, bboxes, image_shape, min_ratio=0.25):
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
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.9, label_fields=['labels']))

        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        if len(transformed['bboxes']) == len(bboxes):
            return transformed['image'], transformed['bboxes'], transformed['labels']
        return None, None, None

    def _save_xml(self, root_template, bboxes, labels, xml_path):
        root = ET.Element("annotation")
        for child in root_template:
            if child.tag != "object":
                root.append(child)
        size = root.find("size") or ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "800"
        ET.SubElement(size, "height").text = "800"
        for label, bbox in zip(labels, bboxes):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            bnd = ET.SubElement(obj, "bndbox")
            for name, val in zip(["xmin","ymin","xmax","ymax"], bbox):
                ET.SubElement(bnd, name).text = str(int(val))
        ET.ElementTree(root).write(xml_path)

    def process_image(self, img_path, xml_path, out_img_dir, out_xml_dir, base_name):
        image = cv2.imread(img_path)
        if image is None:
            return 0
        h, w = image.shape[:2]
        tree = ET.parse(xml_path)
        root_template = tree.getroot()

        bboxes, labels = [], []
        for obj in root_template.findall("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            bboxes.append([float(b.find("xmin").text), float(b.find("ymin").text),
                           float(b.find("xmax").text), float(b.find("ymax").text)])
            labels.append(name)

        if not bboxes:
            return 0

        generated = 0
        is_large = max(w, h) >= self.large_threshold

        # 1. Full
        full_img = A.Resize(800, 800)(image=image)['image']
        cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_full.jpg"), full_img)
        self._save_xml(root_template, bboxes, labels, os.path.join(out_xml_dir, f"{base_name}_full.xml"))
        generated += 1

        if not is_large:
            return generated

        # 2. Zoom global
        z_img, z_bb, z_lb = self._crop_and_resize(image, bboxes, labels, margin=1.5)
        if z_img is not None:
            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_zoom.jpg"), z_img)
            self._save_xml(root_template, z_bb, z_lb, os.path.join(out_xml_dir, f"{base_name}_zoom.xml"))
            generated += 1

        # 3. Split si clusters détectés
        clusters = self._detect_clusters(bboxes, image.shape)
        if clusters:
            idx1, idx2 = clusters
            for idx, suffix in zip([idx1, idx2], ["top", "bottom"]):
                g_bboxes = [bboxes[i] for i in idx]
                g_labels = [labels[i] for i in idx]
                c_img, c_bb, c_lb = self._crop_and_resize(image, g_bboxes, g_labels, margin=1.8)
                if c_img is not None:
                    cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_{suffix}.jpg"), c_img)
                    self._save_xml(root_template, c_bb, c_lb, os.path.join(out_xml_dir, f"{base_name}_{suffix}.xml"))
                    generated += 1

        return generated


def resize_entire_dataset(root_dataset_path: str):
    resizer = UltimateSmartResizer(large_threshold=1200)

    output_root = os.path.join(root_dataset_path, "resized_ultimate")
    splits = ['train', 'val', 'test']

    total_in, total_out = 0, 0

    for split in splits:
        src_img = os.path.join(root_dataset_path, split, "images")
        src_ann = os.path.join(root_dataset_path, split, "Annotations")
        if not (os.path.exists(src_img) and os.path.exists(src_ann)):
            print(f"→ {split} non trouvé")
            continue

        out_img = os.path.join(output_root, split, "images")
        out_ann = os.path.join(output_root, split, "Annotations")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_ann, exist_ok=True)

        print(f"\nTraitement de {split} → {len(os.listdir(src_img))} images")

        for img_file in os.listdir(src_img):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            total_in += 1
            base = os.path.splitext(img_file)[0]
            img_path = os.path.join(src_img, img_file)
            xml_path = os.path.join(src_ann, base + ".xml")
            if not os.path.exists(xml_path):
                continue

            gen = resizer.process_image(img_path, xml_path, out_img, out_ann, base)
            total_out += gen
            print(f"  {img_file} → {gen} versions")

    print("\n" + "="*60)
    print("TOUT EST FINI BORIS !")
    print(f"Images d'origine : {total_in}")
    print(f"Images générées  : {total_out} (+{total_out - total_in} bonus)")
    print(f"Dossier sortie → {output_root}")
    print("="*60)


# ======================== LANCE ÇA ========================
if __name__ == "__main__":
    # METS JUSTE TON DOSSIER RACINE ICI
    root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc"

    resize_entire_dataset(root)