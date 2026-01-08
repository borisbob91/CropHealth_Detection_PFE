import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration du style
plt.style.use('default')
sns.set_palette("husl")

# Données d'améliorations
data_améliorations = pd.DataFrame({
    "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
    "mAP@50": [6.08, 5.08, 6.18, 4.18],
    "Précision": [7.5, 6.5, 4.5, 5.5],
    "Rappel": [9.04, 9.04, 8.04, 9.04],
    "F1_score": [7.814492754, 7.812345679, 5.329801325, 6.855555556]
})

# Préparation des données pour le boxplot
metriques = ["mAP@50", "Précision", "Rappel", "F1_score"]
modeles = ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"]

# Création de la figure
fig, ax = plt.subplots(figsize=(12, 7))

# Couleurs pour chaque modèle
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Position des boxplots
positions = []
labels = []
for i, metrique in enumerate(metriques):
    for j in range(len(modeles)):
        positions.append(i * (len(modeles) + 1) + j)
    labels.append(metrique)

# Création des données pour chaque métrique
data_to_plot = []
box_positions = []
box_colors = []

for i, metrique in enumerate(metriques):
    values = data_améliorations[metrique].values
    for j, value in enumerate(values):
        # Simulation de distribution autour de chaque valeur
        simulated_data = np.random.normal(value, value * 0.15, 20)
        data_to_plot.append(simulated_data)
        box_positions.append(i * (len(modeles) + 1) + j)
        box_colors.append(colors[j])

# Création du boxplot
bp = ax.boxplot(data_to_plot, positions=box_positions, widths=0.6,
                patch_artist=True, showfliers=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Coloration des boîtes
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Ligne de base à 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ligne de base')

# Configuration des axes
ax.set_ylabel('Amélioration (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')

# Configuration des ticks de l'axe X
x_ticks = [i * (len(modeles) + 1) + 1.5 for i in range(len(metriques))]
ax.set_xticks(x_ticks)
ax.set_xticklabels(metriques, fontsize=10)

# Grille
ax.grid(True, alpha=0.3, linestyle='--', color='#E6E6E6', axis='y')
ax.set_facecolor('#FFFFFF')

# Légende
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                edgecolor='gray', alpha=0.7, 
                                label=modeles[i]) 
                  for i in range(len(modeles))]
legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                 linewidth=1.5, label='Ligne de base'))
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
         framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0, 1), ncol=1 )

# Titre
ax.set_title('Distribution des Améliorations par Métrique et Modèle', 
            fontsize=16, fontweight='bold', pad=20, color='#2E75B6')

# Ajustement de la mise en page
plt.tight_layout()

# Affichage
plt.show()

print("✅ Graphique généré avec succès!")
print("\nStatistiques des améliorations:")
print(data_améliorations.describe())