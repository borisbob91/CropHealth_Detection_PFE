import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# Demander le fichier Excel à l'utilisateur
fichier_excel = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\state\instances_augment.xlsx"

# Lire les données depuis Excel
try:
    df = pd.read_excel(fichier_excel)
    print("\nDonnées chargées avec succès!")
    print(df.head())
    
except FileNotFoundError:
   pass

# TRI DÉCROISSANT par nombre d'objets
# df = df.sort_values('total_objets', ascending=False)

# Extraire les données triées
classes = df['Classe'].tolist()
total_objets = df['total_objets'].tolist()
total_img = df['total_img'].tolist()

# Calculer les totaux pour les pourcentages
total_objets_global = sum(total_objets)
total_img_global = sum(total_img)

# Calculer les pourcentages
pourcentage_objets = [obj/total_objets_global * 100 for obj in total_objets]
pourcentage_img = [img/total_img_global * 100 for img in total_img]

# ============================================
# VERSION 1 : STYLE BARRES GROUPÉES (POURCENTAGES) - TRI DÉCROISSANT
# ============================================

fig1, ax1 = plt.subplots(figsize=(18, 10))

x = np.arange(len(classes))
largeur = 0.35

# Création des barres groupées
bars1 = ax1.bar(x - largeur/2, pourcentage_objets, largeur, 
                label='% Total Objets', color='#2E86AB', edgecolor='black', 
                linewidth=1.5, zorder=3, alpha=0.9)
bars2 = ax1.bar(x + largeur/2, pourcentage_img, largeur, 
                label='% Total Images', color='#A23B72', edgecolor='black', 
                linewidth=1.5, zorder=3, alpha=0.9)


ax1.set_ylabel('Pourcentage (%)', fontsize=12, )
ax1.set_xlabel('Classes', fontsize=12,)
ax1.set_xticks(x)

# Étiquettes de l'axe X avec rotation
plt.setp(ax1.get_xticklabels(), rotation=14, ha='right', fontsize=11)
ax1.set_xticklabels(classes)

ax1.set_ylim(0, max(pourcentage_objets + pourcentage_img) * 1.15)

# Format pourcentage sur l'axe Y
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Fonction pour placer les textes intelligemment
def ajouter_textes_pourcentages(bars, valeurs_pct, valeurs_abs, ax, est_objets=True):
    """Ajoute les textes sans chevauchement pour les pourcentages"""
    positions_utilisees = []
    
    for bar, pct, abs_val in zip(bars, valeurs_pct, valeurs_abs):
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        
        # Texte avec pourcentage uniquement
        texte = f"{pct:.1f}%"
        
        # Position verticale
        y_pos = height + 0.3
        
        # Vérifier les chevauchements
        chevauchement = False
        for pos in positions_utilisees:
            if abs(pos[0] - x_pos) < 0.25 and abs(pos[1] - y_pos) < 1.0:
                chevauchement = True
                break
        
        # Ajuster la position si chevauchement
        if chevauchement:
            y_pos = height + 0.8
        
        # Couleur du texte adaptée
        color = '#2E86AB' if est_objets else '#A23B72'
        
        # Ajouter le texte
        if height > 0.5:
            ax.text(x_pos, y_pos, texte, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor=color, alpha=0.9, linewidth=1))
        
        positions_utilisees.append((x_pos, y_pos))

# Ajouter les textes
ajouter_textes_pourcentages(bars1, pourcentage_objets, total_objets, ax1, est_objets=True)
ajouter_textes_pourcentages(bars2, pourcentage_img, total_img, ax1, est_objets=False)

# Légende en haut
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fontsize=14, framealpha=0.9)

# Ajuster la disposition pour éviter le chevauchement avec les étiquettes
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
