import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Demander le fichier Excel à l'utilisateur
fichier_excel = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\state\instances_augment.xlsx"

# Lire les données depuis Excel
try:
    df = pd.read_excel(fichier_excel)
    print("\nDonnées chargées avec succès!")
    print(df.head())
    
except FileNotFoundError:
    print(f"\nErreur: Le fichier '{fichier_excel}' n'a pas été trouvé.")
    print("Création de données de démonstration...")
    
    # Données de démonstration
    data = {
        'Classe': ['A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp',
                  'Earias spp', 'Effet phyto', 'G. spodoctera', 'H. amirgera', 'Jasside',
                  'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron',
                  'S. derogata', 'S. frugiperda', 'Scarabees'],
        'total_objets': [51, 176, 297, 332, 110, 355, 237, 58, 398, 1797, 372, 61, 67, 1186, 655, 222, 95],
        'total_img': [51, 30, 282, 152, 113, 354, 205, 58, 398, 199, 266, 35, 67, 276, 552, 223, 92]
    }
    df = pd.DataFrame(data)

# TRI DÉCROISSANT par nombre d'images
#df = df.sort_values('total_img', ascending=False)

# Extraire les données triées
classes = df['Classe'].tolist()
total_img = df['total_img'].tolist()
total_objets = df['total_objets'].tolist()

# ============================================
# GRAPHIQUE VERTICAL AVEC INSTANCES DANS LES BARRES
# ============================================

# Créer une figure avec une taille optimale
plt.figure(figsize=(22, 12))

x = np.arange(len(classes))
largeur = 0.65

# Couleur pour les barres d'images
couleur_barres = '#A23B72'

# Création des barres (hauteur = nombre d'images)
bars = plt.bar(x, total_img, width=largeur, color=couleur_barres, 
               edgecolor='black', linewidth=2.5, alpha=0.85, zorder=3)

# Personnalisation du titre et des axes
#plt.title('Distribution du Nombre d\'Images par Classe', 
#         fontsize=26, fontweight='bold', pad=35, loc='center')
plt.ylabel('Nombre d\'Images', fontsize=12, fontweight='bold', labelpad=18)
plt.xlabel('Classes\n', fontsize=12, fontweight='bold', labelpad=18)

# Étiquettes de l'axe X avec rotation et taille augmentée
plt.xticks(x, classes, rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=12)

# Ajuster les limites de l'axe Y
y_max = max(total_img)
plt.ylim(0, y_max * 1.25)

# Grille horizontale discrète
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0, linewidth=0.8)

# AJOUTER LE NOMBRE D'INSTANCES DANS LE CORPS DES BARRES
for bar, nb_images, nb_instances in zip(bars, total_img, total_objets):
    height = bar.get_height()
    x_center = bar.get_x() + bar.get_width() / 2
    
    # Texte avec le nombre d'instances À L'INTÉRIEUR de la barre
    if height > y_max * 0.1:  # Si la barre est assez haute
        plt.text(x_center, height * 0.5,
                f'{nb_instances:,}' if nb_instances >= 1000 else str(nb_instances),
                ha='center', va='center',
                fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor='#333333', 
                         edgecolor='white', 
                         alpha=0.95, 
                         linewidth=2))
    else:  # Petite barre
        plt.text(x_center, height * 0.5,
                str(nb_instances),
                ha='center', va='center',
                fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='#333333', 
                         edgecolor='white', 
                         alpha=0.8, 
                         linewidth=1))
    
    # Ajouter le nombre d'images AU-DESSUS de la barre
    plt.text(x_center, height + y_max * 0.02,
            f'{nb_images:,}' if nb_images >= 1000 else str(nb_images),
            ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor='#A23B72', 
                     alpha=0.9, 
                     linewidth=1.5))

"""
# Ajouter une ligne horizontale pour la moyenne
moyenne = np.mean(total_img)
plt.axhline(y=moyenne, color='#2E86AB', linestyle='--', linewidth=3, 
            alpha=0.8, zorder=2)

# Annotation pour la moyenne

plt.annotate(f'Moyenne: {moyenne:.0f} images', 
             xy=(len(classes)-0.8, moyenne * 1.05),
             xytext=(len(classes)-3, moyenne * 1.2),
             fontsize=17, fontweight='bold', color='#2E86AB',
             arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
             bbox=dict(boxstyle='round,pad=0.4', 
                      facecolor='white', 
                      edgecolor='#2E86AB', 
                      alpha=0.95))
"""


# Formater l'axe Y avec séparateurs de milliers
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Ajouter un peu d'espace autour des barres
plt.xlim(-0.7, len(classes) - 0.3)

# Ajustement PRÉCIS de l'espace pour tout voir
plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.28)

# Ajouter une légende explicative
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=couleur_barres, alpha=0.85, edgecolor='black',
          label='Hauteur = Nombre d\'images'),
    Patch(facecolor='#333333', alpha=0.95, edgecolor='white',
          label='Texte interne = Nombre d\'instances')
]
plt.legend(handles=legend_elements, loc='upper center', fontsize=14, 
           framealpha=0.95, borderpad=1)

plt.show()

# ============================================
# STATISTIQUES DÉTAILLÉES
# ============================================

print(f"\n{'='*100}")
print("DISTRIBUTION DES IMAGES ET INSTANCES PAR CLASSE (Tri décroissant par images)")
print(f"{'='*100}")
print(f"{'#':<3} {'CLASSE':<25} {'IMAGES':>10} {'INSTANCES':>12} {'RATIO I/Im':>12} {'% IMAGES':>10}")
print(f"{'-'*100}")

total_images = sum(total_img)
total_instances = sum(total_objets)

for i, (classe, img, obj) in enumerate(zip(classes, total_img, total_objets), 1):
    pct_img = img / total_images * 100
    ratio = obj / img if img > 0 else 0
    
    # Mettre en évidence les classes importantes
    if pct_img > 10:
        highlight = "***"
    elif pct_img > 5:
        highlight = "**"
    elif pct_img > 2:
        highlight = "*"
    else:
        highlight = ""
    
    print(f"{i:<3} {classe:<25}{highlight} {img:>10,} {obj:>12,} {ratio:>11.2f} {pct_img:>9.1f}%")

print(f"{'-'*100}")
print(f"{'TOTAL':<28} {total_images:>10,} {total_instances:>12,} {total_instances/total_images:>11.2f}")
print(f"{'='*100}")

# Calculer quelques statistiques
moyenne_img = np.mean(total_img)
moyenne_inst = np.mean(total_objets)
ratio_global = total_instances / total_images

print(f"\n📊 STATISTIQUES DESCRIPTIVES:")
print(f"  • Total images: {total_images:,}")
print(f"  • Total instances: {total_instances:,}")
print(f"  • Nombre de classes: {len(classes)}")
print(f"  • Moyenne images/classe: {moyenne_img:.1f}")
print(f"  • Moyenne instances/classe: {moyenne_inst:.1f}")
print(f"  • Ratio global instances/images: {ratio_global:.2f}")

print(f"\n🏆 TOP 5 DES CLASSES PAR NOMBRE D'IMAGES:")
for i in range(min(5, len(classes))):
    pct = (total_img[i] / total_images) * 100
    ratio = total_objets[i] / total_img[i]
    print(f"  {i+1}. {classes[i]}: {total_img[i]:,} images, {total_objets[i]:,} instances (ratio: {ratio:.2f})")

print(f"\n📈 CLASSES AVEC LE PLUS D'INSTANCES PAR IMAGE:")
ratios = [(classes[i], total_objets[i]/total_img[i], total_img[i], total_objets[i]) 
          for i in range(len(classes))]
ratios_sorted = sorted(ratios, key=lambda x: x[1], reverse=True)
for i in range(min(5, len(ratios_sorted))):
    classe, ratio, imgs, objs = ratios_sorted[i]
    print(f"  {i+1}. {classe}: {ratio:.2f} instances/image ({objs:,} inst., {imgs:,} img.)")

print(f"{'='*100}")