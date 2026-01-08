import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Données des instances
instances_data = {
    'Classe': ['Jasside', 'Puceron', 'S. derogata', 'H. amirgera', 'Larve coccinelle', 
               'Degat Jassides', 'Earias spp', 'Coccinelle', 'Effet phyto', 'S. frugiperda',
               'Dysdercus spp', 'B. tabaci', 'Scarabees', 'P. gossypiella', 'Larve syrphe',
               'G. spodoctera', 'A. flava'],
    'total': [3585, 2342, 1241, 784, 744, 664, 633, 587, 474, 415, 216, 182, 144, 134, 117, 116, 98],
    'code': ['JA', 'AG', 'SD', 'HA', 'COL', 'JAT', 'ES', 'CO', 'PH', 'SF', 'DY', 'BT', 'SC', 'PG', 'LS', 'GSP', 'AF']
}

# Données des images (correspondance avec les codes)
images_data = {
    #'Code': ['SD', 'HA', 'ES', 'CO', 'AG', 'COL', 'SF', 'PH', 'JA', 'JAT', 'DY', 'SC', 'PG', 'GSP', 'AF', 'LS', 'BT'],
    'Classe_img': ['IMG_R_SD', 'IMG_R_HA', 'IMG_R_ES', 'IMG_P_CO', 'IMG_R_AG', 'IMG_P_COL', 
                   'IMG_R_SF', 'IMG_E_PH', 'IMG_R_JA', 'IMG_R_JAT', 'IMG_R_DS', 'IMG_N_SC', 
                   'IMG_R_PG', 'IMG_R_GSP', 'IMG_R_AF', 'IMG_P_LS', 'IMG_R_BT'],
    'total_img': [1039, 784, 632, 557, 544, 532, 417, 410, 395, 304, 220, 138, 134, 116, 98, 65, 36]
}

instances_data_df = pd.DataFrame(images_data)


df_instances = pd.DataFrame(instances_data)
df_instances = df_instances.sort_values('total', ascending=False)

# Graphique 1 : Instances avec noms complets
plt.figure(figsize=(14, 8))

bars = plt.bar(df_instances['Classe'], df_instances['total'], 
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(df_instances))),
               edgecolor='black', linewidth=1.2)

plt.title('Répartition des Instances par Classe - Noms complets', 
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Nombre d\'Instances', fontsize=14, fontweight='bold')
plt.xlabel('Classes', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Ajout des valeurs
for bar, valeur in zip(bars, df_instances['total']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 30,
            f'{valeur:,}', ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Ajout du code entre parenthèses
for i, (classe, code) in enumerate(zip(df_instances['Classe'], df_instances['code'])):
    plt.text(i, -150, f'({code})', ha='center', va='top', 
             fontsize=10, fontweight='bold', color='darkred')

plt.ylim(0, df_instances['total'].max() * 1.15)
plt.tight_layout()
plt.show()

