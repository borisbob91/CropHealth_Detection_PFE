import argparse
import os
import xml.etree.ElementTree as ET
import shutil

def nettoyer_dataset_ultimate(root_path: str, supprimer_images_vides=True, backup=True):
    """
    Nettoie TON dataset de A à Z :
    1. XML sans image → supprimé
    2. Image sans XML → supprimé
    3. XML avec 0 objet (pas de <object>) → supprimé
    4. Si supprimer_images_vides=True → supprime aussi l'image correspondante
    """
    splits = ['train', 'val', 'test']
    stats = {
        'xml_sans_image': 0,
        'img_sans_xml': 0,
        'xml_vides': 0,
        'img_vides_supprimees': 0
    }

    if backup:
        backup_dir = root_path + "_BACKUP_ULTIMATE"
        if not os.path.exists(backup_dir):
            shutil.copytree(root_path, backup_dir)
            print(f"Backup complet créé → {backup_dir}\n")

    for split in splits:
        img_dir = os.path.join(root_path, split, 'images')
        ann_dir = os.path.join(root_path, split, 'Annotations')

        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            print(f"→ {split} manquant")
            continue

        print(f"\nNettoyage de {split}...")

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG')

        # 1. Lister toutes les images existantes
        images_existantes = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
                             if f.lower().endswith(image_extensions)}

        # 2. Parcourir tous les XML
        for xml_file in os.listdir(ann_dir):
            if not xml_file.lower().endswith('.xml'):
                continue

            base_name = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(ann_dir, xml_file)
            img_paths = [os.path.join(img_dir, base_name + ext) for ext in image_extensions]

            # Cas 1 : XML sans image
            if not any(os.path.exists(p) for p in img_paths):
                os.remove(xml_path)
                print(f"   Supprimé XML sans image → {xml_file}")
                stats['xml_sans_image'] += 1
                continue

            # Cas 2 : XML vide (0 <object>)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objects = root.findall('object')
                if len(objects) == 0:
                    os.remove(xml_path)
                    print(f"   Supprimé XML VIDE (0 objet) → {xml_file}")
                    stats['xml_vides'] += 1

                    if supprimer_images_vides:
                        for img_path in img_paths:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                print(f"      → Image correspondante supprimée : {os.path.basename(img_path)}")
                                stats['img_vides_supprimees'] += 1
                    continue
            except:
                print(f"   XML corrompu → {xml_file} (supprimé)")
                os.remove(xml_path)
                stats['xml_vides'] += 1
                continue

        # 3. Supprimer les images sans XML (après avoir nettoyé les XML)
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(image_extensions):
                base = os.path.splitext(img_file)[0]
                xml_path = os.path.join(ann_dir, base + '.xml')
                if not os.path.exists(xml_path):
                    img_path = os.path.join(img_dir, img_file)
                    os.remove(img_path)
                    print(f"   Supprimée image sans XML → {img_file}")
                    stats['img_sans_xml'] += 1

    # Résumé final
    print("\n" + "="*60)
    print(" NETTOYAGE ULTIMATE TERMINÉ !")
    print("="*60)
    print(f"XML sans image          → {stats['xml_sans_image']}")
    print(f"Images sans XML         → {stats['img_sans_xml']}")
    print(f"XML vides (0 objet)     → {stats['xml_vides']}")
    if supprimer_images_vides:
        print(f"Images vides supprimées → {stats['img_vides_supprimees']}")
    print("="*60)
    print("Ton dataset est maintenant PARFAIT : 1 image = 1 XML avec au moins 1 objet ✅")

# ======================== À LANCER ========================
if __name__ == "__main__":
    dataset_root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc_ac"
    ## ajouter helper argparse et usage

    parser = argparse.ArgumentParser(description="Nettoyer un dataset Pascal VOC")
    parser.add_argument('-p', "--path", type=str, default=dataset_root, help="Chemin vers le dossier racine du dataset")
    parser.add_argument('-d', "--delete-empty", default=True, action="store_true", help="Supprimer les images correspondant à des XML vides")
    parser.add_argument('-b', "--backup", action="store_true", help="Faire une sauvegarde avant nettoyage")
    args = parser.parse_args()
    dataset_root = args.path
    # Première fois : backup + supprime aussi les images vides
    nettoyer_dataset_ultimate(dataset_root, supprimer_images_vides=args.delete_empty, backup=args.backup)