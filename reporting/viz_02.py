import matplotlib.pyplot as plt
import numpy as np

# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
precision_augmente = [77.5, 88.5, 81.5, 79.5]
rappel_augmente = [77.04, 89.04, 82.04, 79.04]

# Configuration du graphique
plt.figure(figsize=(12, 7))

x = np.arange(len(modeles))
largeur = 0.35

# Création des barres groupées
bars1 = plt.bar(x - largeur/2, precision_augmente, largeur, 
                label='Précision', color='#2E86AB', edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + largeur/2, rappel_augmente, largeur, 
                label='Rappel', color='#A23B72', edgecolor='black', linewidth=1.5)

# Personnalisation
plt.title('Précision et Rappel - Données Augmentées', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Valeur (%)', fontsize=16, fontweight='bold')
plt.xlabel('Modèles', fontsize=16, fontweight='bold')
plt.xticks(x, modeles, fontsize=14)
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres (Précision)
for bar, valeur in zip(bars1, precision_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{valeur:.1f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Valeurs sur les barres (Rappel)
for bar, valeur in zip(bars2, rappel_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{valeur:.1f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Légende en haut, centrée
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fontsize=14, framealpha=0.9)

# Ajustements finaux
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
map_augmente = [78.08, 89.08, 85.18, 80.18]

# Configuration du graphique
plt.figure(figsize=(10, 7))

# Couleurs pour chaque modèle
couleurs = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Création des barres
bars = plt.bar(modeles, map_augmente, color=couleurs, 
               edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

# Personnalisation
plt.title('mAP@50 - Données Augmentées', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('mAP@50 (%)', fontsize=16, fontweight='bold')
plt.xlabel('Modèles', fontsize=16, fontweight='bold')
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres
for bar, valeur in zip(bars, map_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'{valeur:.1f}', ha='center', va='top',
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

# Ajustements finaux
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
f1_augmente = [76.8, 88.8, 80.8, 78.8]

# Configuration du graphique
plt.figure(figsize=(10, 7))

# Couleurs pour chaque modèle
couleurs = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Création des barres
bars = plt.bar(modeles, f1_augmente, color=couleurs, 
               edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

# Personnalisation
plt.title('F1-score - Données Augmentées', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('F1-score (%)', fontsize=16, fontweight='bold')
plt.xlabel('Modèles', fontsize=16, fontweight='bold')
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres
for bar, valeur in zip(bars, f1_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'{valeur:.1f}', ha='center', va='top',
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

# Ajustements finaux
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


