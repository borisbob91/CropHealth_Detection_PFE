import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Garder l'ordre original des classes
classes_ordre_original = data_classes['Classe'].tolist()

# ============================================
# GRAPHIQUE 1 : PRÉCISION ET RAPPEL PAR CLASSE
# ============================================

fig1 = go.Figure()

# Ajouter les barres pour la Précision
fig1.add_trace(go.Bar(
    x=data_classes['Classe'],
    y=data_classes['Précision'] * 100,  # Convertir en pourcentage
    name='Précision',
    marker_color='#2E86AB',  # Bleu comme dans ton style
    text=[f'{p*100:.1f}%' for p in data_classes['Précision']],
    textposition='outside',
    textfont=dict(size=12, color='#2E86AB', family='Arial, sans-serif'),
    hovertemplate='<b>%{x}</b><br>Précision: %{y:.1f}%<extra></extra>',
    offsetgroup=1
))

# Ajouter les barres pour le Rappel
fig1.add_trace(go.Bar(
    x=data_classes['Classe'],
    y=data_classes['Rappel'] * 100,  # Convertir en pourcentage
    name='Rappel',
    marker_color='#A23B72',  # Rose comme dans ton style
    text=[f'{r*100:.1f}%' for r in data_classes['Rappel']],
    textposition='outside',
    textfont=dict(size=12, color='#A23B72', family='Arial, sans-serif'),
    hovertemplate='<b>%{x}</b><br>Rappel: %{y:.1f}%<extra></extra>',
    offsetgroup=2
))

# Mise en page
fig1.update_layout(
    title=dict(
        text="<b>Précision et Rappel par Classe</b>",
        font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=40, b=20)
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
        categoryarray=classes_ordre_original,  # Garder l'ordre original
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    yaxis=dict(
        title=dict(
            text="<b>Valeur (%)</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        ticksuffix="%",
        range=[0, 105],  # De 0% à 105% pour laisser de l'espace
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.05,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=16, family='Arial, sans-serif'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#cccccc',
        borderwidth=1
    ),
    height=700,
    width=1400,
    margin=dict(l=80, r=50, t=120, b=180),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

# Ligne horizontale à 80% pour référence
fig1.add_shape(
    type="line",
    x0=-0.5,
    y0=80,
    x1=len(data_classes)-0.5,
    y1=80,
    line=dict(
        color="#FF6B6B",
        width=2,
        dash="dash",
    )
)

# Annotation pour la ligne de référence
fig1.add_annotation(
    x=len(data_classes)-1,
    y=82,
    text="Seuil: 80%",
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
# GRAPHIQUE 2 : mAP@50 PAR CLASSE
# ============================================

fig2 = go.Figure()

# Palette de couleurs pour chaque classe
colors_map = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D8C7B',
    '#8A4F7D', '#3B8EA5', '#E4572E', '#17BEBB', '#6B2737',
    '#3D5A80', '#EE6C4D', '#293241', '#98C1D9', '#E0FBFC',
    '#9B5DE5', '#00BBF9'
]

# Ajouter les barres pour mAP@50
fig2.add_trace(go.Bar(
    x=data_classes['Classe'],
    y=data_classes['mAP@50'] * 100,  # Convertir en pourcentage
    name='mAP@50',
    marker_color=colors_map[:len(data_classes)],
    text=[f'{m*100:.1f}%' for m in data_classes['mAP@50']],
    textposition='outside',
    textfont=dict(size=13, color='white', family='Arial, sans-serif', weight='bold'),
    hovertemplate='<b>%{x}</b><br>mAP@50: %{y:.1f}%<extra></extra>',
    marker=dict(
        line=dict(
            color='#333333',
            width=2
        )
    )
))

# Mise en page
fig2.update_layout(
    title=dict(
        text="<b>mAP@50 par Classe</b>",
        font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=40, b=20)
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
        categoryarray=classes_ordre_original,  # Garder l'ordre original
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    yaxis=dict(
        title=dict(
            text="<b>mAP@50 (%)</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        ticksuffix="%",
        range=[0, 105],
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1
    ),
    showlegend=False,
    height=700,
    width=1400,
    margin=dict(l=80, r=50, t=120, b=180),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

# Ligne horizontale à 80% pour référence
fig2.add_shape(
    type="line",
    x0=-0.5,
    y0=80,
    x1=len(data_classes)-0.5,
    y1=80,
    line=dict(
        color="#FF6B6B",
        width=2,
        dash="dash",
    )
)

fig2.show()

# ============================================
# GRAPHIQUE 3 : F1-SCORE PAR CLASSE
# ============================================

fig3 = go.Figure()

# Palette de couleurs différente pour F1-Score
colors_f1 = [
    '#2E86AB', '#3B8EA5', '#4A96B0', '#59A0BA', '#68AAC4',
    '#77B4CE', '#86BED8', '#95C8E2', '#A4D2EC', '#B3DCF6',
    '#C2E6FF', '#D1F0FF', '#E0FAFF', '#EFFAFF', '#FEFFFF',
    '#EDF7FF', '#DCEFFF'
]

# Ajouter les barres pour F1-Score
fig3.add_trace(go.Bar(
    x=data_classes['Classe'],
    y=data_classes['F1_score'] * 100,  # Convertir en pourcentage
    name='F1-Score',
    marker_color=colors_f1[:len(data_classes)],
    text=[f'{f*100:.1f}%' for f in data_classes['F1_score']],
    textposition='outside',
    textfont=dict(size=13, color='white', family='Arial, sans-serif', weight='bold'),
    hovertemplate='<b>%{x}</b><br>F1-Score: %{y:.1f}%<extra></extra>',
    marker=dict(
        line=dict(
            color='#333333',
            width=2
        )
    )
))

# Mise en page
fig3.update_layout(
    title=dict(
        text="<b>F1-Score par Classe</b>",
        font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        x=0.5,
        xanchor='center',
        pad=dict(t=40, b=20)
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
        categoryarray=classes_ordre_original,  # Garder l'ordre original
        showgrid=False,
        linecolor='#cccccc',
        linewidth=1
    ),
    yaxis=dict(
        title=dict(
            text="<b>F1-Score (%)</b>",
            font=dict(size=18, family='Arial, sans-serif', color='#333333')
        ),
        tickfont=dict(size=14, family='Arial, sans-serif'),
        ticksuffix="%",
        range=[0, 105],
        showgrid=True,
        gridcolor='#e0e0e0',
        gridwidth=1,
        linecolor='#cccccc',
        linewidth=1
    ),
    showlegend=False,
    height=700,
    width=1400,
    margin=dict(l=80, r=50, t=120, b=180),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='Arial, sans-serif'
    )
)

# Ligne horizontale à 80% pour référence
fig3.add_shape(
    type="line",
    x0=-0.5,
    y0=80,
    x1=len(data_classes)-0.5,
    y1=80,
    line=dict(
        color="#FF6B6B",
        width=2,
        dash="dash",
    )
)

fig3.show()

# ============================================
# TABLEAU RÉCAPITULATIF DES PERFORMANCES
# ============================================

print(f"\n{'='*120}")
print("PERFORMANCES PAR CLASSE - TABLEAU RÉCAPITULATIF")
print(f"{'='*120}")
print(f"{'Classe':<20} {'Précision':>12} {'Rappel':>12} {'mAP@50':>12} {'F1-Score':>12} {'Statut':>15}")
print(f"{'-'*120}")

for idx, row in data_classes.iterrows():
    precision = row['Précision'] * 100
    rappel = row['Rappel'] * 100
    map50 = row['mAP@50'] * 100
    f1 = row['F1_score'] * 100
    
    # Déterminer le statut global
    scores = [precision, rappel, map50, f1]
    moyenne = sum(scores) / len(scores)
    
    if moyenne >= 90:
        statut = "✅ EXCELLENT"
        statut_color = "🟢"
    elif moyenne >= 80:
        statut = "🟡 BON"
        statut_color = "🟡"
    elif moyenne >= 70:
        statut = "🟠 MOYEN"
        statut_color = "🟠"
    elif moyenne >= 60:
        statut = "🔴 FAIBLE"
        statut_color = "🔴"
    else:
        statut = "❌ CRITIQUE"
        statut_color = "❌"
    
    print(f"{row['Classe']:<20} {precision:>11.1f}% {rappel:>11.1f}% {map50:>11.1f}% "
          f"{f1:>11.1f}% {statut_color:>2} {statut:<12}")

print(f"{'-'*120}")

# Calcul des moyennes
moy_precision = data_classes['Précision'].mean() * 100
moy_rappel = data_classes['Rappel'].mean() * 100
moy_map50 = data_classes['mAP@50'].mean() * 100
moy_f1 = data_classes['F1_score'].mean() * 100

print(f"{'MOYENNE':<20} {moy_precision:>11.1f}% {moy_rappel:>11.1f}% {moy_map50:>11.1f}% "
      f"{moy_f1:>11.1f}%")
print(f"{'='*120}")

# ============================================
# STATISTIQUES DÉTAILLÉES
# ============================================

print(f"\n📊 ANALYSE DES PERFORMANCES PAR CLASSE")
print(f"{'='*60}")

# Meilleures performances
print(f"\n🏆 TOP 3 MEILLEURES PERFORMANCES:")
for metric in ['Précision', 'Rappel', 'mAP@50', 'F1_score']:
    top3 = data_classes.nlargest(3, metric)
    print(f"\n  {metric}:")
    for idx, row in top3.iterrows():
        value = row[metric] * 100
        print(f"    • {row['Classe']}: {value:.1f}%")

# Performances critiques
print(f"\n⚠️ CLASSES AVEC PERFORMANCES FAIBLES (< 70%):")
faibles = data_classes[
    (data_classes['Précision'] < 0.7) | 
    (data_classes['Rappel'] < 0.7) | 
    (data_classes['mAP@50'] < 0.7) |
    (data_classes['F1_score'] < 0.7)
]

if not faibles.empty:
    for idx, row in faibles.iterrows():
        print(f"\n  {row['Classe']}:")
        metrics_low = []
        if row['Précision'] < 0.7:
            metrics_low.append(f"Précision: {row['Précision']*100:.1f}%")
        if row['Rappel'] < 0.7:
            metrics_low.append(f"Rappel: {row['Rappel']*100:.1f}%")
        if row['mAP@50'] < 0.7:
            metrics_low.append(f"mAP@50: {row['mAP@50']*100:.1f}%")
        if row['F1_score'] < 0.7:
            metrics_low.append(f"F1-Score: {row['F1_score']*100:.1f}%")
        
        for metric in metrics_low:
            print(f"    • {metric}")
else:
    print("  Aucune classe avec performances critiques.")

print(f"\n📈 CLASSES AVEC PERFORMANCES EXCELLENTES (> 95%):")
excellentes = data_classes[
    (data_classes['Précision'] > 0.95) & 
    (data_classes['Rappel'] > 0.95) & 
    (data_classes['mAP@50'] > 0.95) &
    (data_classes['F1_score'] > 0.95)
]

if not excellentes.empty:
    for idx, row in excellentes.iterrows():
        print(f"  • {row['Classe']}:")
        print(f"      Précision: {row['Précision']*100:.1f}%")
        print(f"      Rappel: {row['Rappel']*100:.1f}%")
        print(f"      mAP@50: {row['mAP@50']*100:.1f}%")
        print(f"      F1-Score: {row['F1_score']*100:.1f}%")
else:
    print("  Aucune classe avec toutes les métriques > 95%")

print(f"{'='*60}")

# Sauvegarder les graphiques
fig1.write_html("precision_rappel_par_classe.html")
fig2.write_html("map50_par_classe.html")
fig3.write_html("f1_score_par_classe.html")

print(f"\n💾 Graphiques sauvegardés:")
print("  • precision_rappel_par_classe.html")
print("  • map50_par_classe.html")
print("  • f1_score_par_classe.html")