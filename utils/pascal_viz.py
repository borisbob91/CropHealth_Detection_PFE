import os
import cv2
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys

# Defaults (can be overridden with CLI args or by calling run())
default_dataset = r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\train'
default_num_samples = 6  # Nombre d'images à afficher par défaut

# 📄 Fonction pour lire les annotations VOC
def read_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return bboxes, labels

# 🎯 Fonction d'affichage
def show_random_images_with_boxes(images_dir, labels_dir, num_samples=5):
    # Récupérer toutes les images
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(all_images) == 0:
        print("❌ Aucune image trouvée dans le dossier images/")
        return
    
    # Sélectionner des échantillons aléatoires
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    print(f"📸 Affichage de {len(samples)} images...\n")
    
    for i, img_file in enumerate(samples, 1):
        img_path = os.path.join(images_dir, img_file)
        xml_file = img_file.rsplit('.', 1)[0] + '.xml'
        xml_path = os.path.join(labels_dir, xml_file)
        
        # Vérifier que le fichier XML existe
        if not os.path.exists(xml_path):
            print(f"⚠️ Fichier XML manquant pour {img_file} (cherché: {xml_file})")
            continue
        
        # Lire l'image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Impossible de lire l'image {img_file}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lire les annotations
        try:
            bboxes, labels_list = read_voc_annotation(xml_path)
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {xml_file}: {e}")
            continue
        
        # Créer la figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Dessiner les boîtes
        for bbox, label in zip(bboxes, labels_list):
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Ajouter le label
            ax.text(xmin, ymin - 5, label, color='white', fontsize=12,
                    weight='bold', bbox=dict(facecolor='red', alpha=0.7))
        
        ax.set_title(f"Image {i}/{len(samples)}: {img_file}", fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"✅ {img_file} - {len(bboxes)} objet(s) détecté(s): {', '.join(labels_list)}")

def run(dataset: str = None, num_images: int = None):
    """Helper to run visualization from script or notebook.

    - `dataset` should be a folder containing `images/` and `Annotations/` (VOC XMLs).
    - `num_images` is how many random images to display.
    """
    if dataset is None:
        dataset = default_dataset
    if num_images is None:
        num_images = default_num_samples

    images_dir = os.path.join(dataset, 'images')
    labels_dir = os.path.join(dataset, 'Annotations')

    # If running inside IPython (notebook/Colab), enable inline matplotlib
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic('matplotlib', 'inline')
    except Exception:
        # ignore if IPython not available
        pass

    show_random_images_with_boxes(images_dir, labels_dir, num_images)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Visualiser des images avec boîtes VOC (XML).')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=default_dataset,
        help=r"Chemin du dossier (contenant images/ et Annotations/)"
    )
    parser.add_argument(
        '--num_images', '-n',
        type=int,
        default=default_num_samples,
        help="Nombre d'images aléatoires à visualiser (défaut: 6)"
    )
    return parser


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()
    run(dataset=args.dataset, num_images=args.num_images)