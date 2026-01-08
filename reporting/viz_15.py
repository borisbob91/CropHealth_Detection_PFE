import pandas as pd
import plotly.graph_objects as go

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

# TRI DÉCROISSANT par nombre d'images
df = df.sort_values('total_img', ascending=False)

# Calculer le pourcentage d'images
total_img_global = df['total_img'].sum()
df['pourcentage_img'] = df['total_img'] / total_img_global * 100

# ============================================
# VERSION 1 : GRAPHIQUE BARRES SIMPLES (IMAGES SEULEMENT)
# ============================================

fig1 = go.Figure()

# Ajouter les barres pour les images
fig1.add_trace(go.Bar(
    x=df['Classe'],
    y=df['total_img'],
    name='Nombre d\'Images',
    marker_color='#A23B72',  # Rose comme dans tes styles originaux
    text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_img']],
    textposition='outside',
    textfont=dict(size=12, color='#A23B72', family='Arial, sans-serif'),
    hovertemplate='<b>%{x}</b><br>Images: %{y:,}<br>Pourcentage: %{customdata:.1f}%<extra></extra>',
    customdata=df['pourcentage_img']
))

# Mise en page
fig1.update_layout(
    title=dict(
        text="<b>Nombre d'Images par Classe</b><br><i>(Tri décroissant par nombre d'images)</i>",
        font=dict(size=22, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=30, b=20)
    ),
    xaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        tickangle=45,
        type='category',
        categoryorder='array',
        categoryarray=df['Classe'].tolist(),
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    yaxis=dict(
        title=dict(
            text="<b>Nombre d'Images</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        tickformat=',',
        range=[0, df['total_img'].max() * 1.15],
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1,
        zeroline=True,
        zerolinecolor='#cccccc',
        zerolinewidth=1
    ),
    showlegend=False,
    height=700,
    width=1200,
    margin=dict(l=100, r=50, t=150, b=180),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

# Ajouter une ligne horizontale pour la moyenne
moyenne_img = df['total_img'].mean()
fig1.add_shape(
    type="line",
    x0=-0.5,
    y0=moyenne_img,
    x1=len(df)-0.5,
    y1=moyenne_img,
    line=dict(
        color="#FF6B6B",
        width=2,
        dash="dash",
    )
)

# Ajouter une annotation pour la moyenne
fig1.add_annotation(
    x=len(df)-1,
    y=moyenne_img * 1.05,
    text=f"Moyenne: {moyenne_img:.0f} images",
    showarrow=False,
    font=dict(size=12, color="#FF6B6B", family='Arial, sans-serif'),
    bgcolor="white",
    bordercolor="#FF6B6B",
    borderwidth=1,
    borderpad=4,
    opacity=0.9
)

fig1.show()

# ============================================
# VERSION 2 : GRAPHIQUE EN POURCENTAGES (IMAGES SEULEMENT)
# ============================================

fig2 = go.Figure()

# Palette de couleurs pour chaque barre
colors = ['#A23B72', '#C73E1D', '#F18F01', '#2E86AB', '#5D8C7B',
          '#8A4F7D', '#3B8EA5', '#E4572E', '#17BEBB', '#6B2737',
          '#3D5A80', '#EE6C4D', '#293241', '#98C1D9', '#E0FBFC',
          '#9B5DE5', '#00BBF9']

# Ajouter les barres pour les pourcentages d'images
fig2.add_trace(go.Bar(
    x=df['Classe'],
    y=df['pourcentage_img'],
    name='% d\'Images',
    marker_color=colors[:len(df)],
    text=[f'{p:.1f}%' for p in df['pourcentage_img']],
    textposition='outside',
    textfont=dict(size=12, color='white', family='Arial, sans-serif'),
    hovertemplate='<b>%{x}</b><br>Pourcentage: %{y:.1f}%<br>Nombre: %{customdata:,} images<extra></extra>',
    customdata=df['total_img'],
    marker=dict(
        line=dict(
            color='#333333',
            width=1.5
        )
    )
))

# Mise en page
fig2.update_layout(
    title=dict(
        text="<b>Distribution en Pourcentage des Images par Classe</b>",
        font=dict(size=22, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=30, b=20)
    ),
    xaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        tickangle=45,
        type='category',
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    yaxis=dict(
        title=dict(
            text="<b>Pourcentage d'Images (%)</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        ticksuffix="%",
        range=[0, df['pourcentage_img'].max() * 1.15],
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1
    ),
    showlegend=False,
    height=700,
    width=1200,
    margin=dict(l=100, r=50, t=150, b=180),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

fig2.show()

# ============================================
# VERSION 3 : GRAPHIQUE HORIZONTAL (ALTERNATIVE)
# ============================================

fig3 = go.Figure()

# Graphique horizontal pour une meilleure lisibilité des noms
fig3.add_trace(go.Bar(
    y=df['Classe'],
    x=df['total_img'],
    name='Nombre d\'Images',
    orientation='h',
    marker_color='#2E86AB',  # Bleu comme dans tes styles originaux
    text=[f'{v:,}' if v >= 1000 else str(v) for v in df['total_img']],
    textposition='outside',
    textfont=dict(size=12, color='#2E86AB', family='Arial, sans-serif'),
    hovertemplate='<b>%{y}</b><br>Images: %{x:,}<br>Pourcentage: %{customdata:.1f}%<extra></extra>',
    customdata=df['pourcentage_img']
))

# Mise en page du graphique horizontal
fig3.update_layout(
    title=dict(
        text="<b>Nombre d'Images par Classe (Graphique Horizontal)</b>",
        font=dict(size=22, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=30, b=20)
    ),
    yaxis=dict(
        title=dict(
            text="<b>Classes</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        autorange="reversed",  # Pour avoir la plus grande valeur en haut
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    xaxis=dict(
        title=dict(
            text="<b>Nombre d'Images</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        tickformat=',',
        range=[0, df['total_img'].max() * 1.1],
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1
    ),
    showlegend=False,
    height=800,  # Plus haut pour accommoder les noms de classes
    width=1000,
    margin=dict(l=180, r=100, t=150, b=100),  # Marge gauche plus grande pour les noms longs
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

fig3.show()

# ============================================
# STATISTIQUES DÉTAILLÉES
# ============================================

print(f"\n{'='*80}")
print("STATISTIQUES DES IMAGES PAR CLASSE:")
print(f={'='*80})
print(f"{'Classe':<20} {'Images':>12} {'Pourcentage':>12} {'Cumul %':>12}")
print(f"{'-'*80}")

cumul_pct = 0
for i, (_, row) in enumerate(df.iterrows(), 1):
    cumul_pct += row['pourcentage_img']
    print(f"{i:2d}. {row['Classe']:<17} {row['total_img']:>12,} "
          f"{row['pourcentage_img']:>11.1f}% {cumul_pct:>11.1f}%")

print(f"{'-'*80}")
print(f"{'TOTAL':<20} {total_img_global:>12,} {100:>11.1f}%")
print(f"{'='*80}")

# Statistiques résumées
print(f"\n{'='*80}")
print("RÉSUMÉ DES STATISTIQUES:")
print(f={'='*80})
print(f"Nombre total d'images: {total_img_global:,}")
print(f"Nombre moyen d'images par classe: {df['total_img'].mean():.1f}")
print(f"Médiane d'images par classe: {df['total_img'].median():.1f}")
print(f"Écart-type: {df['total_img'].std():.1f}")
print(f"Minimum: {df['total_img'].min()} ({df.loc[df['total_img'].idxmin(), 'Classe']})")
print(f"Maximum: {df['total_img'].max():,} ({df.loc[df['total_img'].idxmax(), 'Classe']})")
print(f"\nClasses avec plus de 100 images ({len(df[df['total_img'] > 100])} classes):")
for _, row in df[df['total_img'] > 100].iterrows():
    print(f"  • {row['Classe']}: {row['total_img']:,} images ({row['pourcentage_img']:.1f}%)")
print(f"{'='*80}")

# Sauvegarder les graphiques
fig1.write_html("images_par_classe.html")
fig2.write_html("pourcentage_images_par_classe.html")
fig3.write_html("images_par_classe_horizontal.html")

print("\nGraphiques sauvegardés:")
print("  • images_par_classe.html")
print("  • pourcentage_images_par_classe.html")
print("  • images_par_classe_horizontal.html")