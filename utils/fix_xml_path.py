import os
import xml.etree.ElementTree as ET

def corriger_filename_dans_xml(root_dataset_path: str):
    """
    Corrige le champ <filename> dans tous les XML pour qu'il corresponde
    au vrai nom du fichier image (ex: 000f247f-IMG_R_HA_109_zoom.jpg)
    """
    splits = ['train', 'val', 'test']
    total_corriges = 0

    print("Correction des <filename> dans les XML...\n")

    for split in splits:
        ann_dir = os.path.join(root_dataset_path, split, 'Annotations')
        img_dir = os.path.join(root_dataset_path, split, 'images')

        if not os.path.exists(ann_dir):
            print(f"→ {split}/Annotations non trouvé")
            continue
        if not os.path.exists(img_dir):
            print(f"→ {split}/images non trouvé")
            continue

        print(f"Traitement de {split}...")

        for xml_file in os.listdir(ann_dir):
            if not xml_file.lower().endswith('.xml'):
                continue

            xml_path = os.path.join(ann_dir, xml_file)
            base_name = os.path.splitext(xml_file)[0]  # ex: 000f247f-IMG_R_HA_109_zoom

            # Trouver l'image correspondante (peut être .jpg ou .png)
            possible_images = [f for f in os.listdir(img_dir) 
                               if f.startswith(base_name) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not possible_images:
                continue  # pas d'image → on touche pas

            vrai_nom_image = possible_images[0]  # ex: 000f247f-IMG_R_HA_109_zoom.jpg

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename_tag = root.find('filename')
                if filename_tag is None:
                    # Si pas de <filename>, on le crée
                    filename_tag = ET.SubElement(root, 'filename')

                ancien = filename_tag.text
                if ancien != vrai_nom_image:
                    filename_tag.text = vrai_nom_image
                    tree.write(xml_path)
                    total_corriges += 1
                    print(f"   Corrigé : {xml_file}  →  {vrai_nom_image}")

            except Exception as e:
                print(f"   Erreur sur {xml_file} : {e}")

    print("\n" + "="*60)
    print(f"CORRECTION TERMINÉE ! {total_corriges} fichiers XML mis à jour.")
    print("Tous les <filename> correspondent maintenant au vrai nom de l'image")
    print("="*60)


# ======================== À LANCER ========================
if __name__ == "__main__":
    # METS TON DOSSIER RACINE ICI (celui avec train/val/test)
    root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc_ac"

    corriger_filename_dans_xml(root)