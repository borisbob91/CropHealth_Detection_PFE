import matplotlib.pyplot as plt
import numpy as np

# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
f1_original = [68.9855, 80.9877, 75.4702, 71.9444]
f1_augmente = [76.8, 88.8, 80.8, 78.8]

x = np.arange(len(modeles))
largeur = 0.35

# Création du graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Barres
rects1 = ax.bar(x - largeur/2, f1_original, largeur, label='Données Originales', color='steelblue', edgecolor='black')
rects2 = ax.bar(x + largeur/2, f1_augmente, largeur, label='Données Augmentées', color='lightcoral', edgecolor='black')

# Labels et titres
ax.set_xlabel('Modèles', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-score (%)', fontsize=14, fontweight='bold')
ax.set_title('Comparaison du F1-score : Données Originales vs. Augmentées', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modeles, fontsize=12)
ax.set_ylim(0, 100)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ajout des valeurs sur les barres
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # décalage vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
autolabel(rects1)
autolabel(rects2)

# Ajustement layout
plt.tight_layout()
plt.show()