import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Données des performances par classe
data_classes = pd.DataFrame({
    "Classe": [
        "A. flava", "B. tabaci", "Coccinelle", "Degat Jassides", "Dysdercus spp",
        "Earias spp", "Effet phyto", "G. spodoptera", "H. armigera", "Jasside",
        "Larve coccinelle", "Larve syrphe", "P. gossypiella", "Puceron",
        "S. derogata", "S. frugiperda", "Scarabees"
    ],
    "Précision": [
        0.98, 0.964, 0.814, 0.788, 0.978,
        0.981, 0.603, 0.996, 0.986, 0.855,
        0.941, 0.959, 0.925, 0.505,
        0.905, 0.933, 0.995
    ],
    "Rappel": [
        0.987, 0.941, 0.872, 0.682, 0.98,
        0.966, 0.738, 1.000, 0.989, 0.805,
        0.991, 1.000, 0.967, 0.360,
        0.940, 1.000, 0.880
    ],
    "mAP@50": [
        0.978, 0.966, 0.870, 0.739, 0.974,
        0.992, 0.698, 0.995, 0.994, 0.807,
        0.992, 0.995, 0.969, 0.386,
        0.920, 0.984, 0.886
    ],
    "F1_score": [
        0.983487544, 0.952361155, 0.842002372, 0.731178231, 0.978998979,
        0.973442219, 0.663704698, 0.997995992, 0.987497722, 0.824515152,
        0.965353002, 0.979070955, 0.945533827, 0.420346821,
        0.922168022, 0.965338852, 0.933973333
    ]
})

# Trier par mAP@50 décroissant
data_sorted = data_classes.sort_values('mAP@50', ascending=False)
classes = data_sorted['Classe'].tolist()
map_scores = (data_sorted['mAP@50'] * 100).tolist()  # Conversion en pourcentage

# Configuration du graphique
plt.figure(figsize=(16, 10))

x = np.arange(len(classes))
bar_width = 0.7

# Créer un dégradé de couleurs basé sur les scores
# Du vert (haut score) au orange (bas score)
colors = plt.cm.RdYlGn((np.array(map_scores) - min(map_scores)) / (max(map_scores) - min(map_scores)))

# Création des barres
bars = plt.bar(x, map_scores, bar_width,
               color=colors,
               edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

# Personnalisation
plt.title('Performance mAP@50 par Classe d\'Insectes', 
          fontsize=24, fontweight='bold', pad=30)
plt.ylabel('mAP@50 (%)', fontsize=20, fontweight='bold')
plt.xlabel('Classes d\'Insectes', fontsize=20, fontweight='bold')

plt.xticks(x, classes, fontsize=13, rotation=45, ha='right')
plt.ylim(30, 105)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres avec couleur adaptée
for bar, score in zip(bars, map_scores):
    height = bar.get_height()
    
    # Ajuster la couleur du texte selon la luminosité
    if score > 85:  # Scores élevés - texte blanc
        text_color = 'white'
        bg_color = '#2E8B57'  # Vert foncé
    elif score > 70:  # Scores moyens - texte sombre
        text_color = '#333333'
        bg_color = '#FFD700'  # Or
    else:  # Scores bas - texte blanc
        text_color = 'white'
        bg_color = '#DC143C'  # Rouge
    
    plt.text(bar.get_x() + bar.get_width()/2., height - 3,
             f'{score:.1f}', ha='center', va='top',
             fontsize=12, fontweight='bold', color=text_color,
             bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor=bg_color, 
                      alpha=0.9, edgecolor='black'))

# Ligne horizontale pour la moyenne
mean_map = np.mean(map_scores)
plt.axhline(y=mean_map, color='#2E86AB', linestyle='--', 
            linewidth=3, alpha=0.8, zorder=1)

# Annotation de la moyenne
plt.text(len(classes) - 0.5, mean_map + 2,
         f'Moyenne: {mean_map:.1f}%',
         fontsize=14, fontweight='bold', color='#2E86AB',
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', 
                  facecolor='white', 
                  edgecolor='#2E86AB', alpha=0.9))

# Ligne horizontale pour le seuil de 90%
plt.axhline(y=90, color='#2E8B57', linestyle=':', 
            linewidth=2, alpha=0.6, zorder=1)
plt.text(0, 91, 'Seuil 90%',
         fontsize=11, fontweight='bold', color='#2E8B57',
         ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor='#2E8B57', alpha=0.8))

# Mettre en évidence les 3 meilleures classes
top_n = 3
for i in range(top_n):
    plt.annotate(f'{i+1}°', 
                 xy=(i, map_scores[i]),
                 xytext=(i, map_scores[i] + 5),
                 ha='center', va='bottom',
                 fontsize=14, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gold'),
                 bbox=dict(boxstyle='circle,pad=0.3', 
                          facecolor='gold', 
                          alpha=0.8, edgecolor='darkgoldenrod'))

# Mettre en évidence les 3 moins bonnes classes
bottom_n = 3
for i in range(1, bottom_n + 1):
    idx = len(map_scores) - i
    plt.annotate('⚠', 
                 xy=(idx, map_scores[idx]),
                 xytext=(idx, map_scores[idx] - 8),
                 ha='center', va='top',
                 fontsize=14, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                 bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='lightcoral', 
                          alpha=0.8, edgecolor='red'))

# Ajouter une légende de couleur
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#2E8B57', alpha=0.9, label='Score > 85% (Excellent)'),
    Patch(facecolor='#FFD700', alpha=0.9, label='Score 70-85% (Bon)'),
    Patch(facecolor='#DC143C', alpha=0.9, label='Score < 70% (À améliorer)'),
    Patch(facecolor='white', edgecolor='#2E86AB', linestyle='--', 
          linewidth=2, label=f'Moyenne: {mean_map:.1f}%')
]

plt.legend(handles=legend_elements, 
           loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=4, fontsize=13, framealpha=0.9, frameon=True)

# Ajustements finaux
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Espace pour la légende

plt.show()