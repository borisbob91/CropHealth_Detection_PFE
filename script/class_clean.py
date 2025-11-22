import os
import xml.etree.ElementTree as ET
import shutil

def supprimer_classes_intelligente(
    root_path: str,
    classes_a_supprimer: list,
    backup: bool = True
):
    """
    Nettoyage INTELLIGENT :
    - Si classe indésirable + autres objets → supprime SEULEMENT l'objet indésirable
    - Si classe indésirable est SEULE → supprime l'image + XML complet
    """
    classes_indesirables = [c.strip().lower() for c in classes_a_supprimer]
    splits = ['train', 'val', 'test']
    
    stats = {
        'objets_supprimes': 0,
        'images_supprimees': 0,
        'images_nettoyees': 0
    }
    details = []

    if backup:
        backup_dir = root_path + "_BACKUP_NETTOYAGE_INTELLIGENT"
        if not os.path.exists(backup_dir):
            shutil.copytree(root_path, backup_dir)
            print(f"Backup créé → {backup_dir}\n")

    print(f"Nettoyage intelligent des classes : {', '.join(classes_a_supprimer)}\n")

    for split in splits:
        img_dir = os.path.join(root_path, split, 'images')
        ann_dir = os.path.join(root_path, split, 'Annotations')

        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            continue

        print(f"Traitement {split}...")

        for xml_file in os.listdir(ann_dir):
            if not xml_file.lower().endswith('.xml'):
                continue

            xml_path = os.path.join(ann_dir, xml_file)
            base_name = os.path.splitext(xml_file)[0]

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objects = root.findall('object')

                objets_indesirables = []
                objets_valides = []

                for obj in objects:
                    class_name = obj.find('name').text.strip().lower()
                    if class_name in classes_indesirables:
                        objets_indesirables.append(obj)
                    else:
                        objets_valides.append(obj)

                # === CAS 1 : Il reste des objets valides → on garde l'image, on vire juste les indésirables ===
                if objets_valides and objets_indesirables:
                    for obj in objets_indesirables:
                        root.remove(obj)
                        stats['objets_supprimes'] += 1
                    tree.write(xml_path)
                    stats['images_nettoyees'] += 1
                    details.append(f"{split}/{base_name} → {len(objets_indesirables)} objet(s) supprimé(s), image conservée")

                # === CAS 2 : SEULEMENT des objets indésirables → on vire TOUT ===
                elif not objets_valides and objets_indesirables:
                    # Supprimer XML
                    os.remove(xml_path)
                    
                    # Supprimer toutes les images possibles
                    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
                    for ext in extensions:
                        img_path = os.path.join(img_dir, base_name + ext)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    
                    stats['images_supprimees'] += 1
                    stats['objets_supprimes'] += len(objets_indesirables)
                    details.append(f"{split}/{base_name} → image + XML supprimés (seulement classes indésirables)")

            except Exception as e:
                print(f"Erreur {xml_file}: {e}")

    # Résumé final
    print("\n" + "="*70)
    print(" NETTOYAGE INTELLIGENT TERMINÉ !")
    print("="*70)
    print(f"Objets indésirables supprimés     → {stats['objets_supprimes']}")
    print(f"Images entièrement supprimées     → {stats['images_supprimees']}")
    print(f"Images nettoyées (objets gardés)  → {stats['images_nettoyees']}")
    print("="*70)
    if details:
        print("Exemples :")
        for d in details[:15]:
            print(f"   • {d}")
        if len(details) > 15:
            print(f"   ... et {len(details)-15} autres")
    print("Ton dataset est maintenant PARFAIT et sans perte inutile !")

# ======================== À LANCER ========================
if __name__ == "__main__":
    dataset_root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc"

    classes_a_virer = [
        'Fourmie',
        'punaisse_01',
        'punaisse_02',
        'punaisse_03',
        'punaisse_04',
        'Syrphe',
        'Degat A. flava',
    ]

    supprimer_classes_intelligente(
        root_path=dataset_root,
        classes_a_supprimer=classes_a_virer,
        backup=True
    )