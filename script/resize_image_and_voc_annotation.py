import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A

def resize_image_and_voc_annotation(image_path, xml_path, output_image_path, output_xml_path, target_size=(800, 800)):
    """
    Redimensionne une image et ses annotations XML Pascal VOC.
    """
    try:
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur: Impossible de charger l'image {image_path}")
            return False
        
        orig_height, orig_width = image.shape[:2]

        # Lire l'annotation XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extraire les boxes et labels
        bboxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # Définir la transformation
        transform = A.Compose([
            A.Resize(height=target_size[1], width=target_size[0])
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        # Appliquer la transformation
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        resized_image = transformed['image']
        resized_bboxes = transformed['bboxes']

        # Vérification de cohérence
        if len(resized_bboxes) != len(bboxes):
            print(f"Attention: Nombre de boxes différent pour {image_path}")
            return False

        # Sauvegarder l'image redimensionnée
        cv2.imwrite(output_image_path, resized_image)

        # Mettre à jour l'annotation XML
        root.find("size").find("width").text = str(target_size[0])
        root.find("size").find("height").text = str(target_size[1])

        # Mettre à jour chaque bbox
        for obj, new_bbox in zip(root.findall("object"), resized_bboxes):
            bbox = obj.find("bndbox")
            bbox.find("xmin").text = str(int(new_bbox[0]))
            bbox.find("ymin").text = str(int(new_bbox[1]))
            bbox.find("xmax").text = str(int(new_bbox[2]))
            bbox.find("ymax").text = str(int(new_bbox[3]))

        # Sauvegarder le nouveau XML
        tree.write(output_xml_path)
        
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {str(e)}")
        return False


if __name__ == "__main__":
    # Configuration
    root_dir = r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\train'
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'Annotations')
    
    output_root = os.path.join(root_dir, 'resized_dataset')
    output_images_dir = os.path.join(output_root, 'images')
    output_labels_dir = os.path.join(output_root, 'Annotations')
    
    # Créer les dossiers de sortie
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Paramètres
    target_size = (800, 800)
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Compteurs
    total_files = 0
    success_files = 0
    
    print("Début du redimensionnement...")
    
    # Parcourir toutes les images
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(supported_formats):
            total_files += 1
            
            # Chemins d'entrée
            image_path = os.path.join(images_dir, filename)
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(labels_dir, xml_filename)
            
            # Vérifier que le XML existe
            if not os.path.exists(xml_path):
                print(f"XML manquant pour {filename} -> {xml_path}")
                continue
            
            # Chemins de sortie
            output_image_path = os.path.join(output_images_dir, filename)
            output_xml_path = os.path.join(output_labels_dir, xml_filename)
            
            # Redimensionner
            if resize_image_and_voc_annotation(image_path, xml_path, output_image_path, output_xml_path, target_size):
                success_files += 1
                print(f"[{success_files}/{total_files}] ✓ {filename}")
    
    print(f"\n{'='*50}")
    print(f"Traitement terminé !")
    print(f"Fichiers traités avec succès: {success_files}/{total_files}")
    print(f"Dossier de sortie: {output_root}")
    print(f"{'='*50}")