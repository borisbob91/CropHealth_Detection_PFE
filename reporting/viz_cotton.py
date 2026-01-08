import matplotlib.pyplot as plt
import numpy as np

# Données
campagnes = ['2022-2023', '2023-2024']
production = [236186, 347922]  # en tonnes

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Création de l'histogramme
bars = ax.bar(campagnes, production, color=['skyblue', 'lightgreen'], edgecolor='black', linewidth=1.5, width=0.6)

# Ajout des valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
            f'{height:,} t', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Personnalisation du graphique
ax.set_title('Production de coton en Côte d\'Ivoire\nCampagnes 2022-2023 vs 2023-2024', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Production (tonnes)', fontsize=14, fontweight='bold')
ax.set_xlabel('Campagne', fontsize=14, fontweight='bold')

# Calcul et affichage du pourcentage d'augmentation
augmentation = ((production[1] - production[0]) / production[0]) * 100
ax.text(0.5, max(production) * 0.9, 
        f'Augmentation: +{augmentation:.1f}%', 
        ha='center', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Amélioration des ticks
ax.set_ylim(0, max(production) * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Ajout de la source
plt.figtext(0.5, 0.01, 'Source: Communiqué Abidjan, 04 juin 2024', 
            ha='center', fontsize=10, style='italic')

# Ajustement de l'espacement
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Affichage du graphique
plt.show()

# Affichage des données brutes
print("="*50)
print("DONNÉES DE PRODUCTION DE COTON")
print("="*50)
print(f"Campagne 2022-2023: {production[0]:,} tonnes")
print(f"Campagne 2023-2024: {production[1]:,} tonnes")
print(f"Augmentation: {production[1]-production[0]:,} tonnes (+{augmentation:.1f}%)")
print("="*50)