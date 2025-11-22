import os
import xml.etree.ElementTree as ET
import re
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

class DatasetAnalyzer:
    def __init__(self):
        self.pattern_regex = re.compile(r'(?:^|-)([A-Z]+_[A-Z]+_[A-Z]+)(?:_\d+)?')

    def auto_detect_patterns(self, filenames: List[str]) -> Dict[str, str]:
        """Ta fonction exacte  détecte la classe principale depuis le nom"""
        file_to_class = {}
        for filename in filenames:
            basename = os.path.splitext(filename)[0]
            match = self.pattern_regex.search(basename)
            if match:
                file_to_class[filename] = match.group(1)
            else:
                file_to_class[filename] = 'unknown'
        return file_to_class

    def analyze_full_dataset(self, root_path: str, output_prefix: str = "dataset_analysis"):
        """
        Analyse train / val / test avec TA convention
        """
        splits = ['train', 'val', 'test']
        
        # === 1. Comptage images par classe (via nom de fichier) ===
        images_per_class = {split: Counter() for split in splits}
        
        # === 2. Comptage instances par classe (via XML) ===
        instances_per_class = {split: Counter() for split in splits}

        for split in splits:
            img_dir = os.path.join(root_path, split, 'images')
            ann_dir = os.path.join(root_path, split, 'Annotations')

            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                print(f"{split} non trouvé → ignoré")
                continue

            print(f"Analyse {split}...")

            # Lister toutes les images
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Appliquer TA fonction de détection
            file_to_class = self.auto_detect_patterns(image_files)
            
            # Compter les images par classe principale
            for filename, cls in file_to_class.items():
                images_per_class[split][cls] += 1

            # Lire les XML pour les instances réelles
            for xml_file in os.listdir(ann_dir):
                if not xml_file.endswith('.xml'):
                    continue
                xml_path = os.path.join(ann_dir, xml_file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text.strip()
                        instances_per_class[split][class_name] += 1
                except:
                    print(f"Erreur XML: {xml_path}")

        # === Création des DataFrames ===
        all_main_classes = sorted(set().union(*[c.keys() for c in images_per_class.values()]))
        all_instance_classes = sorted(set().union(*[c.keys() for c in instances_per_class.values()]))

        def make_df(counter_dict, classes):
            data = {
                'Classe': classes,
                'train': [counter_dict['train'].get(c, 0) for c in classes],
                'val': [counter_dict['val'].get(c, 0) for c in classes],
                'test': [counter_dict['test'].get(c, 0) for c in classes],
            }
            df = pd.DataFrame(data)
            df['total'] = df['train'] + df['val'] + df['test']
            df['% total'] = (df['total'] / df['total'].sum() * 100).round(2)
            df = df.sort_values('total', ascending=False).reset_index(drop=True)
            return df

        df_images = make_df(images_per_class, all_main_classes)
        df_instances = make_df(instances_per_class, all_instance_classes)

        # Sauvegarde
        img_excel = os.path.join(root_path, f"{output_prefix}_images_par_classe_principale.xlsx")
        inst_excel = os.path.join(root_path, f"{output_prefix}_instances_par_objet.xml.xlsx")
        
        df_images.to_excel(img_excel, index=False)
        df_instances.to_excel(inst_excel, index=False)

        print(f"\nAnalyse terminée !")
        print(f"→ {img_excel}  (images par classe via nom de fichier)")
        print(f"→ {inst_excel}  (instances réelles via XML)")

        return df_images, df_instances

    def plot(self, df_images, df_instances, style='grouped'):
        plt.figure(figsize=(15, 8))
        if style == 'grouped':
            df_plot = df_instances.melt(id_vars='Classe', value_vars=['train','val','test'],
                                        var_name='Split', value_name='Instances')
            sns.barplot(data=df_plot, x='Classe', y='Instances', hue='Split',
                        palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
            plt.title("Répartition des instances par classe (via XML)", fontsize=16, pad=20)
        else:
            df_plot = df_instances.set_index('Classe')[['train','val','test']]
            df_plot.plot(kind='bar', stacked=True, figsize=(15,8),
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            plt.title("Répartition empilée des instances", fontsize=16, pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Nombre d'instances")
        plt.tight_layout()
        plt.savefig(f"plot_{style}.png", dpi=300, bbox_inches='tight')
        plt.show()


# ======================== LANCEMENT ========================
if __name__ == "__main__":
    analyzer = DatasetAnalyzer()

    root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc_ac"

    df_img, df_inst = analyzer.analyze_full_dataset(root, "crop_health_ac")

    # Deux graphiques
    analyzer.plot(df_img, df_inst, style='grouped')
    analyzer.plot(df_img, df_inst, style='stacked')