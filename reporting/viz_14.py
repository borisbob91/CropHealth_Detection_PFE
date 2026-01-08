import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Demander le fichier Excel à l'utilisateur

fichier_excel = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\state\instances_count.xlsx"

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

# TRI DÉCROISSANT par nombre d'objets
df = df.sort_values('total_objets', ascending=False)

# Extraire les données triées
classes = df['Classe'].tolist()
total_objets = df['total_objets'].tolist()
total_img = df['total_img'].tolist()

# Calculer les totaux pour les pourcentages
total_objets_global = sum(total_objets)
total_img_global = sum(total_img)

# Calculer les pourcentages
df['pourcentage_objets'] = df['total_objets'] / total_objets_global * 100
df['pourcentage_img'] = df['total_img'] / total_img_global * 100

# ============================================
# VERSION 1 : GRAPHIQUE EN POURCENTAGES (PLOTLY)
# ============================================

# Créer la figure
fig1 = go.Figure()

# Ajouter les barres pour les objets (pourcentages)
fig1.add_trace(go.Bar(
    x=df['Classe'],
    y=df['pourcentage_objets'],
    name='% Total Objets',
    marker_color='#2E86AB',
    text=[f'{p:.1f}%' for p in df['pourcentage_objets']],
    textposition='outside',
    textfont=dict(size=11, color='#2E86AB'),
    hovertemplate='<b>%{x}</b><br>Objets: %{y:.1f}%<br>Valeur absolue: %{customdata:,}',
    customdata=df['total_objets']
))

# Ajouter les barres pour les images (pourcentages)
fig1.add_trace(go.Bar(
    x=df['Classe'],
    y=df['pourcentage_img'],
    name='% Total Images',
    marker_color='#A23B72',
    text=[f'{p:.1f}%' for p in df['pourcentage_img']],
    textposition='outside',
    textfont=dict(size=11, color='#A23B72'),
    hovertemplate='<b>%{x}</b><br>Images: %{y:.1f}%<br>Valeur absolue: %{customdata:,}',
    customdata=df['total_img']
))

# Mise en page
fig1.update_layout(
    title=dict(
        text="<b>Distribution en Pourcentage - Objets et Images par Classe</b><br><i>(Tri décroissant par nombre d'objets)</i>",
        font=dict(size=20, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=16, family='Arial, sans-serif')
        ),
        tickfont=dict(size=12),
        tickangle=45,
        type='category',
        categoryorder='array',
        categoryarray=classes
    ),
    yaxis=dict(
        title=dict(
            text="<b>Pourcentage (%)</b>",
            font=dict(size=16, family='Arial, sans-serif')
        ),
        tickformat='.0%',
        range=[0, max(df['pourcentage_objets'].max(), df['pourcentage_img'].max()) * 1.15],
        tickfont=dict(size=12)
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=14)
    ),
    height=700,
    margin=dict(l=80, r=50, t=120, b=150),
    plot_bgcolor='white',
    hovermode='x unified'
)

# Ajouter une grille
fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Afficher le graphique
fig1.show()

# ============================================
# VERSION 2 : GRAPHIQUE EN VALEURS ABSOLUES (PLOTLY)
# ============================================

# Créer la figure
fig2 = go.Figure()

# Ajouter les barres pour les objets (valeurs absolues)
fig2.add_trace(go.Bar(
    x=df['Classe'],
    y=df['total_objets'],
    name='Total Objets',
    marker_color='#2E86AB',
    text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_objets']],
    textposition='outside',
    textfont=dict(size=11, color='#2E86AB'),
    hovertemplate='<b>%{x}</b><br>Objets: %{y:,}<br>Pourcentage: %{customdata:.1f}%',
    customdata=df['pourcentage_objets']
))

# Ajouter les barres pour les images (valeurs absolues)
fig2.add_trace(go.Bar(
    x=df['Classe'],
    y=df['total_img'],
    name='Total Images',
    marker_color='#A23B72',
    text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_img']],
    textposition='outside',
    textfont=dict(size=11, color='#A23B72'),
    hovertemplate='<b>%{x}</b><br>Images: %{y:,}<br>Pourcentage: %{customdata:.1f}%',
    customdata=df['pourcentage_img']
))

# Mise en page
fig2.update_layout(
    title=dict(
        text="<b>Distribution des Objets et Images par Classe - Valeurs Absolues</b><br><i>(Tri décroissant par nombre d'objets)</i>",
        font=dict(size=20, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=16, family='Arial, sans-serif')
        ),
        tickfont=dict(size=12),
        tickangle=45,
        type='category',
        categoryorder='array',
        categoryarray=classes
    ),
    yaxis=dict(
        title=dict(
            text="<b>Nombre</b>",
            font=dict(size=16, family='Arial, sans-serif')
        ),
        range=[0, max(df['total_objets'].max(), df['total_img'].max()) * 1.1],
        tickfont=dict(size=12),
        tickformat=','
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=14)
    ),
    height=700,
    margin=dict(l=80, r=50, t=120, b=150),
    plot_bgcolor='white',
    hovermode='x unified'
)

# Ajouter une grille
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Afficher le graphique
fig2.show()

# ============================================
# VERSION 3 : GRAPHIQUE COMBINÉ (FACULTATIF)
# Un graphique avec deux axes Y
# ============================================

fig3 = make_subplots(specs=[[{"secondary_y": True}]])

# Ajouter les barres pour les objets (axe Y principal)
fig3.add_trace(
    go.Bar(
        x=df['Classe'],
        y=df['total_objets'],
        name="Total Objets",
        marker_color='#2E86AB',
        text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_objets']],
        textposition='outside',
        textfont=dict(size=11, color='#2E86AB'),
        hovertemplate='Objets: %{y:,}<br>%{customdata:.1f}% du total',
        customdata=df['pourcentage_objets']
    ),
    secondary_y=False
)

# Ajouter les barres pour les images (axe Y secondaire)
fig3.add_trace(
    go.Bar(
        x=df['Classe'],
        y=df['total_img'],
        name="Total Images",
        marker_color='#A23B72',
        text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_img']],
        textposition='outside',
        textfont=dict(size=11, color='#A23B72'),
        hovertemplate='Images: %{y:,}<br>%{customdata:.1f}% du total',
        customdata=df['pourcentage_img']
    ),
    secondary_y=True
)

# Mise en page du graphique combiné
fig3.update_layout(
    title=dict(
        text="<b>Distribution Comparée - Objets (gauche) vs Images (droite)</b><br><i>(Tri décroissant par nombre d'objets)</i>",
        font=dict(size=20, family='Arial, sans-serif'),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=16, family='Arial, sans-serif')
        ),
        tickfont=dict(size=12),
        tickangle=45,
        type='category'
    ),
    barmode='group',
    bargap=0.2,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=14)
    ),
    height=700,
    margin=dict(l=80, r=80, t=120, b=150),
    plot_bgcolor='white',
    hovermode='x unified'
)

# Configurer les axes Y
fig3.update_yaxes(
    title_text="<b>Nombre d'Objets</b>",
    secondary_y=False,
    title_font=dict(size=16),
    tickfont=dict(size=12),
    tickformat=',',
    showgrid=True,
    gridcolor='lightgray',
    gridwidth=1
)

fig3.update_yaxes(
    title_text="<b>Nombre d'Images</b>",
    secondary_y=True,
    title_font=dict(size=16),
    tickfont=dict(size=12),
    tickformat=',',
    showgrid=False
)

fig3.show()

# ============================================
# AFFICHAGE DES STATISTIQUES
# ============================================

print(f"\n{'='*70}")
print("CLASSES PAR ORDRE DÉCROISSANT DE NOMBRE D'OBJETS:")
print(f"{'='*70}")
print(f"{'Classe':<20} {'Objets':>10} {'Images':>10} {'% Objets':>10} {'% Images':>10}")
print(f"{'-'*70}")

for i, (_, row) in enumerate(df.iterrows(), 1):
    print(f"{i:2d}. {row['Classe']:<17} {row['total_objets']:>10,} {row['total_img']:>10,} "
          f"{row['pourcentage_objets']:>9.1f}% {row['pourcentage_img']:>9.1f}%")

print(f"{'='*70}")

# Statistiques supplémentaires
print(f"\n{'='*70}")
print("STATISTIQUES GLOBALES:")
print(f"{'='*70}")
print(f"Total objets: {total_objets_global:,}")
print(f"Total images: {total_img_global:,}")
print(f"Ratio moyen objets/image: {total_objets_global/total_img_global:.2f}")
print(f"\nClasses avec déséquilibre majeur (ratio > 5):")
for _, row in df.iterrows():
    ratio = row['total_objets'] / row['total_img'] if row['total_img'] > 0 else 0
    if ratio > 5:
        print(f"  {row['Classe']}: {ratio:.1f} objets/image "
              f"({row['total_objets']:,} objets / {row['total_img']:,} images)")

print(f"{'='*70}")

# Sauvegarder les graphiques (optionnel)
fig1.write_html("graphique_pourcentages.html")
fig2.write_html("graphique_valeurs_absolues.html")
fig3.write_html("graphique_combine.html")
print("\nGraphiques sauvegardés au format HTML dans le répertoire courant.")