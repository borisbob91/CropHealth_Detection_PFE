import matplotlib.pyplot as plt
import numpy as np

# Données
classes = [
    'A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp',
    'Earias spp', 'Effet phyto', 'G. spodoptera', 'H. amirgera', 'Jasside',
    'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron',
    'S. derogata', 'S. frugiperda', 'Scarabee'
]

img_original = [81, 30, 282, 152, 113, 354, 205, 58, 398, 199, 266, 45, 97, 276, 552, 223, 92]
img_aug = [1296, 480, 2538, 1368, 1808, 2124, 1845, 928, 2388, 1791, 2394, 720, 1552, 2484, 3312, 2007, 1472]

# Trier par nombre d'images originales décroissant
sorted_indices = np.argsort(img_original)[::-1]
classes_sorted = [classes[i] for i in sorted_indices]
original_sorted = [img_original[i] for i in sorted_indices]
aug_sorted = [img_aug[i] for i in sorted_indices]

# Configuration du graphique
plt.figure(figsize=(18, 10))

x = np.arange(len(classes_sorted))
largeur = 0.35

# Couleurs de votre charte
couleur_original = '#2E86AB'  # Bleu pour images originales
couleur_aug = '#A23B72'       # Rose pour images augmentées

# Création des barres groupées
bars_original = plt.bar(x - largeur/2, original_sorted, largeur,
                         label='Images Originales', 
                         color=couleur_original, 
                         edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

bars_aug = plt.bar(x + largeur/2, aug_sorted, largeur,
                    label='Images Augmentées', 
                    color=couleur_aug, 
                    edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

# Personnalisation
# plt.title('Comparaison: Images Originales vs Augmentées par Classe', 
#          fontsize=24, fontweight='bold', pad=30)
plt.ylabel('Nombre d\'Images', fontsize=12, fontweight='bold')
plt.xlabel('Classes d\'Insectes', fontsize=12, fontweight='bold')

plt.xticks(x, classes_sorted, fontsize=12, rotation=45, ha='right')
plt.ylim(0, max(aug_sorted) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Échelle logarithmique en Y (optionnel - commentez/décommentez)
# plt.yscale('log')

# Valeurs sur les barres - Images Originales
for bar, valeur in zip(bars_original, original_sorted):
    height = bar.get_height()
    # Pour les petites valeurs, mettre le texte au-dessus
    if height < max(original_sorted) * 0.1:
        y_pos = height + max(aug_sorted) * 0.02
        va_pos = 'bottom'
        color = couleur_original
        bg_color = 'white'
    else:
        y_pos = height - max(aug_sorted) * 0.01
        va_pos = 'top'
        color = 'white'
        bg_color = couleur_original
    
    plt.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{valeur:,}'.replace(',', ' '), ha='center', va=va_pos,
             fontsize=10, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.2', 
                      facecolor=bg_color, 
                      alpha=0.9, edgecolor='black'))

# Valeurs sur les barres - Images Augmentées
for bar, valeur in zip(bars_aug, aug_sorted):
    height = bar.get_height()
    # Pour les très grandes valeurs, ajuster la position
    y_pos = height - max(aug_sorted) * 0.02
    va_pos = 'top'
    color = 'white'
    bg_color = couleur_aug
    
    plt.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{valeur:,}'.replace(',', ' '), ha='center', va=va_pos,
             fontsize=10, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.2', 
                      facecolor=bg_color, 
                      alpha=0.9, edgecolor='black'))

# Légende en haut, centrée
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
           ncol=2, fontsize=16, framealpha=0.9, frameon=True)

# Calculer les facteurs d'augmentation
facteurs_aug = [aug/orig if orig > 0 else 0 for orig, aug in zip(original_sorted, aug_sorted)]

# Annoter les facteurs d'augmentation les plus élevés
top_facteurs_idx = np.argsort(facteurs_aug)[-3:][::-1]

for idx in top_facteurs_idx:
    facteur = facteurs_aug[idx]
    if facteur > 20:  # Annoter seulement les forts facteurs
        plt.annotate(f'×{facteur:.0f}', 
                     xy=(x[idx], aug_sorted[idx]),
                     xytext=(x[idx], aug_sorted[idx] + max(aug_sorted)*0.05),
                     ha='center', va='bottom',
                     fontsize=12, fontweight='bold', color='darkgreen',
                     bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor='lightgreen', 
                              alpha=0.8, edgecolor='green'))

# Ajouter une ligne pour la moyenne des images originales
""" 
mean_original = np.mean(original_sorted)
plt.axhline(y=mean_original, color=couleur_original, linestyle='--', 
            linewidth=2, alpha=0.7, zorder=1)
plt.text(len(classes_sorted) - 0.5, mean_original + max(aug_sorted)*0.02,
         f'Moy. originales: {mean_original:.0f}',
         fontsize=10, color=couleur_original,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor=couleur_original, alpha=0.9))
""" 
# Ajouter une ligne pour la moyenne des images augmentées
""" 
mean_aug = np.mean(aug_sorted)
plt.axhline(y=mean_aug, color=couleur_aug, linestyle='--', 
            linewidth=2, alpha=0.7, zorder=1)
plt.text(len(classes_sorted) - 0.5, mean_aug - max(aug_sorted)*0.02,
         f'Moy. augmentées: {mean_aug:.0f}',
         fontsize=10, color=couleur_aug,
         ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor=couleur_aug, alpha=0.9))
""" 
# Ajouter un ratio global
total_original = sum(original_sorted)
total_aug = sum(aug_sorted)
ratio_global = total_aug / total_original

plt.figtext(0.98, 0.98, 
            f'Total originales: {total_original:,}\n'
            f'Total augmentées: {total_aug:,}\n'
            fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='lightblue', 
                     edgecolor='#2E86AB', alpha=0.8))

# Ajouter une mini-barre d'échelle
""" 
plt.figtext(0.02, 0.98, 
            'Échelle:\n'
            'Bleu: Originales\n'
            'Rose: Augmentées\n'
            'Vert: Facteur >10×',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='lightgray', 
                     edgecolor='gray', alpha=0.8),
            verticalalignment='top')
"""


# Ajustements finaux
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.12)

plt.show()


import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Données
classes = [
    'A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp',
    'Earias spp', 'Effet phyto', 'G. spodoptera', 'H. amirgera', 'Jasside',
    'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron',
    'S. derogata', 'S. frugiperda', 'Scarabee'
]

img_original = [81, 30, 282, 152, 113, 354, 205, 58, 398, 199, 266, 45, 97, 276, 552, 223, 92]
img_aug = [1296, 480, 2538, 1368, 1808, 2124, 1845, 928, 2388, 1791, 2394, 720, 1552, 2484, 3312, 2007, 1472]

# Trier par nombre d'images originales décroissant
sorted_indices = np.argsort(img_original)[::-1]
classes_sorted = [classes[i] for i in sorted_indices]
original_sorted = [img_original[i] for i in sorted_indices]
aug_sorted = [img_aug[i] for i in sorted_indices]

# Calculer les facteurs d'augmentation
facteurs_aug = [aug/orig for orig, aug in zip(original_sorted, aug_sorted)]

# Création de la figure
fig = go.Figure()

# Couleurs
couleur_original = '#2E86AB'  # Bleu
couleur_aug = '#A23B72'       # Rose

# Ajouter les barres pour les images originales
fig.add_trace(go.Bar(
    x=classes_sorted,
    y=original_sorted,
    name='Images Originales',
    marker_color=couleur_original,
    marker_line_color='black',
    marker_line_width=1.5,
    opacity=0.9,
    hovertemplate='<b>%{x}</b><br>' +
                  'Originales: %{y:,}<br>' +
                  'Augmentées: %{customdata:,}<br>' +
                  'Facteur: ×%{customdata[1]:.1f}<extra></extra>',
    customdata=np.column_stack([aug_sorted, facteurs_aug])
))

# Ajouter les barres pour les images augmentées
fig.add_trace(go.Bar(
    x=classes_sorted,
    y=aug_sorted,
    name='Images Augmentées',
    marker_color=couleur_aug,
    marker_line_color='black',
    marker_line_width=1.5,
    opacity=0.9,
    hovertemplate='<b>%{x}</b><br>' +
                  'Originales: %{customdata:,}<br>' +
                  'Augmentées: %{y:,}<br>' +
                  'Facteur: ×%{customdata[1]:.1f}<extra></extra>',
    customdata=np.column_stack([original_sorted, facteurs_aug])
))

# Mise en page
fig.update_layout(
    title={
        'text': '<b>Distribution des Images: Originales vs Augmentées</b>',
        'font': {'size': 12, 'family': 'Arial, sans-serif'},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis={
        'title': '<b>Classes d\'Insectes</b>',
        'title_font': {'size': 18},
        'tickfont': {'size': 12},
        'tickangle': 45
    },
    yaxis={
        'title': '<b>Nombre d\'Images</b>',
        'title_font': {'size': 18},
        'tickfont': {'size': 12},
        'gridcolor': 'lightgray',
        'gridwidth': 1,
        'zeroline': True,
        'zerolinecolor': 'gray',
        'zerolinewidth': 1
    },
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font={'family': 'Arial, sans-serif'},
    showlegend=True,
    legend={
        'x': 0.5,
        'y': 1.05,
        'xanchor': 'center',
        'yanchor': 'bottom',
        'orientation': 'h',
        'font': {'size': 14},
        'bgcolor': 'rgba(255, 255, 255, 0.8)',
        'bordercolor': 'black',
        'borderwidth': 1
    },
    height=700,
    width=1200
)

# Ajouter les annotations de valeurs
for i, (orig, aug) in enumerate(zip(original_sorted, aug_sorted)):
    # Annotation pour images originales (si significatif)
    if orig > 50:
        fig.add_annotation(
            x=i,
            y=orig + max(aug_sorted)*0.02,
            text=f"{orig}",
            showarrow=False,
            font={'size': 10, 'color': couleur_original, 'weight': 'bold'},
            yanchor='bottom'
        )
    
    # Annotation pour images augmentées
    fig.add_annotation(
        x=i,
        y=aug + max(aug_sorted)*0.02,
        text=f"{aug:,}".replace(",", " "),
        showarrow=False,
        font={'size': 11, 'color': couleur_aug, 'weight': 'bold'},
        yanchor='bottom',
        bgcolor='white',
        bordercolor=couleur_aug,
        borderwidth=1,
        borderpad=2
    )


# Ajouter des lignes de moyenne
mean_original = np.mean(original_sorted)
mean_aug = np.mean(aug_sorted)

fig.add_hline(
    y=mean_original,
    line_dash="dash",
    line_color=couleur_original,
    line_width=2,
    opacity=0.7,
    annotation_text=f"Moy. originales: {mean_original:.0f}",
    annotation_position="top right",
    annotation_font_size=12,
    annotation_font_color=couleur_original,
    annotation_bgcolor="white",
    annotation_bordercolor=couleur_original
)

fig.add_hline(
    y=mean_aug,
    line_dash="dash",
    line_color=couleur_aug,
    line_width=2,
    opacity=0.7,
    annotation_text=f"Moy. augmentées: {mean_aug:.0f}",
    annotation_position="bottom right",
    annotation_font_size=12,
    annotation_font_color=couleur_aug,
    annotation_bgcolor="white",
    annotation_bordercolor=couleur_aug
)

# Calculer et afficher les totaux
total_original = sum(original_sorted)
total_aug = sum(aug_sorted)
ratio_global = total_aug / total_original

# Ajouter une annotation avec les statistiques globales
fig.add_annotation(
    x=0.02,
    y=0.98,
    xref="paper",
    yref="paper",
    text=f"<b>Statistiques Globales</b><br>"
         f"Originales: {total_original:,}<br>"
         f"Augmentées: {total_aug:,}<br>",
    showarrow=False,
    font={'size': 12, 'family': 'Arial'},
    align='left',
    bgcolor='lightblue',
    bordercolor='#2E86AB',
    borderwidth=2,
    borderpad=8
)


# Personnaliser les tooltips
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>" +
                  "Originales: %{customdata[0]:,}<br>" +
                  "Augmentées: %{y:,}<br>" +
                  "Facteur: ×%{customdata[1]:.1f}<extra></extra>"
)

# Ajouter des boutons pour les interactions
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, True]}],
                    label="Les deux",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False]}],
                    label="Originales seulement",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Augmentées seulement",
                    method="update"
                ),
                dict(
                    args=[{"type": "bar"}, {"barmode": "group"}],
                    label="Groupé",
                    method="update"
                ),
                dict(
                    args=[{"type": "bar"}, {"barmode": "stack"}],
                    label="Empilé",
                    method="update"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Afficher la figure
fig.show()