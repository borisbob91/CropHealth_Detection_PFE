import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger le fichier Excel
file_path = r"c:\Users\BorisBob\Documents\Bob.xlsx"
df = pd.read_excel(file_path, sheet_name="Base Analyse par clone")

print("Structure du fichier Excel:")
print(df.head())
print("\nInformations sur les colonnes:")
print(df.info())

# Trouver les indices des colonnes de données
# D'après la structure montrée, les colonnes D à H contiennent les données de chaque clone
clone_names = []
data_columns = []

# Identifier les colonnes qui contiennent des données numériques
for col in df.columns:
    if df[col].dtype in ['float64', 'int64'] and df[col].notna().any():
        clone_names.append(col)
        data_columns.append(col)
        print(f"Colonne détectée: {col} - {df[col].notna().sum()} valeurs")

# Si nous n'avons pas trouvé de colonnes numériques directement,
# cherchons la première ligne pour les noms de clones
if len(data_columns) == 0:
    print("\nRecherche des noms de clones dans la première ligne...")
    first_row = df.iloc[0]
    for i, value in enumerate(first_row):
        if isinstance(value, str) and value in ['IRCA41', 'IRCA230', 'IRCA331', 'IRCA317', 'RRIC100']:
            clone_names.append(value)
            print(f"Clone trouvé en colonne {i}: {value}")

# Préparation des données par groupe
group1_data = []  # IRCA41, IRCA331, IRCA317
group2_data = []  # RRIC100, IRCA230
group1_labels = []
group2_labels = []

# Collecter les données pour chaque clone
for clone in ['IRCA41', 'IRCA331', 'IRCA317', 'RRIC100', 'IRCA230']:
    clone_data = []
    
    # Essayer différentes méthodes pour trouver les données
    # Méthode 1: Colonne nommée d'après le clone
    if clone in df.columns:
        clone_data = df[clone].dropna().tolist()
    
    # Méthode 2: Chercher dans toutes les colonnes numériques
    elif len(data_columns) > 0:
        for col in data_columns:
            col_data = df[col].dropna().tolist()
            if len(col_data) > 0:
                clone_data = col_data
                break
    
    # Méthode 3: Utiliser la colonne RS filtrée par la colonne CLONE
    elif 'CLONE' in df.columns and 'RS' in df.columns:
        clone_data = df[df['CLONE'] == clone]['RS'].dropna().tolist()
    
    if clone_data:
        print(f"{clone}: {len(clone_data)} valeurs")
        
        if clone in ['IRCA41', 'IRCA331', 'IRCA317']:
            group1_data.append(clone_data)
            group1_labels.append(clone)
        elif clone in ['RRIC100', 'IRCA230']:
            group2_data.append(clone_data)
            group2_labels.append(clone)
    else:
        print(f"Avertissement: Aucune donnée trouvée pour {clone}")

# Si nous n'avons toujours pas de données, explorons la structure autrement
if len(group1_data) == 0 and len(group2_data) == 0:
    print("\nExploration approfondie de la structure...")
    
    # Afficher les premières lignes pour comprendre la structure
    print("\nPremières 10 lignes:")
    print(df.head(10))
    
    # Chercher les valeurs uniques dans la colonne A (CLONE)
    if 'CLONE' in df.columns:
        print("\nValeurs uniques dans CLONE:")
        print(df['CLONE'].unique())
    
    # Si la colonne B est RS, utilisons cette approche
    if df.shape[1] >= 2:
        # La colonne A pourrait être CLONE, B RS
        if df.columns[0] == 'CLONE' and df.columns[1] == 'RS':
            for clone in ['IRCA41', 'IRCA331', 'IRCA317', 'RRIC100', 'IRCA230']:
                mask = df['CLONE'] == clone
                clone_data = df.loc[mask, 'RS'].dropna().tolist()
                if clone_data:
                    print(f"{clone}: {len(clone_data)} valeurs")
                    
                    if clone in ['IRCA41', 'IRCA331', 'IRCA317']:
                        group1_data.append(clone_data)
                        group1_labels.append(clone)
                    elif clone in ['RRIC100', 'IRCA230']:
                        group2_data.append(clone_data)
                        group2_labels.append(clone)

# Vérification finale
print(f"\nRésumé:")
print(f"Groupe 1 ({group1_labels}): {len(group1_data)} ensembles de données")
print(f"Groupe 2 ({group2_labels}): {len(group2_data)} ensembles de données")

if len(group1_data) == 0 or len(group2_data) == 0:
    print("ERREUR: Données insuffisantes pour créer le graphique")
    print("Veuillez vérifier la structure de votre fichier Excel")
else:
    # Création du box plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Positions des box plots
    positions = []
    all_data = []
    colors = []
    labels = []
    
    # Groupe 1 - Même couleur
    group1_color = 'skyblue'
    for i, data in enumerate(group1_data):
        positions.append(i + 1)
        all_data.append(data)
        colors.append(group1_color)
        labels.append(group1_labels[i])
    
    # Groupe 2 - Même couleur (différente)
    group2_color = 'lightcoral'
    for i, data in enumerate(group2_data):
        positions.append(len(group1_data) + i + 2)  # Espace entre groupes
        all_data.append(data)
        colors.append(group2_color)
        labels.append(group2_labels[i])
    
    # Créer les box plots
    box_plots = ax.boxplot(all_data, positions=positions, patch_artist=True, 
                          labels=labels, showmeans=True, meanline=True,
                          meanprops=dict(color='red', linewidth=2),
                          medianprops=dict(color='black', linewidth=2))
    
    # Appliquer les couleurs
    for patch, color in zip(box_plots['boxes'], colors):
        patch.set_facecolor(color)
    
    # Ajouter les valeurs moyennes au-dessus des box plots
    for i, (pos, data) in enumerate(zip(positions, all_data)):
        mean_val = np.mean(data)
        ax.text(pos, ax.get_ylim()[1] * 0.98, f'Moyenne: {mean_val:.3f}', 
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Personnalisation du graphique
    ax.set_title('Box Plot des Clones par Groupe', fontsize=16, fontweight='bold')
    ax.set_ylabel('Valeur RS', fontsize=12)
    ax.set_xlabel('Clones', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group1_color, edgecolor='black', label='Groupe 1: IRCA41, IRCA331, IRCA317'),
        Patch(facecolor=group2_color, edgecolor='black', label='Groupe 2: RRIC100, IRCA230'),
        Patch(facecolor='none', edgecolor='red', label='Moyenne (ligne rouge)'),
        Patch(facecolor='none', edgecolor='black', label='Médiane (ligne noire)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Ajuster les limites pour mieux voir les données
    ax.set_ylim(bottom=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    print("-" * 50)
    
    for i, (label, data) in enumerate(zip(group1_labels + group2_labels, all_data)):
        print(f"\n{label}:")
        print(f"  Nombre d'observations: {len(data)}")
        print(f"  Moyenne: {np.mean(data):.4f}")
        print(f"  Médiane: {np.median(data):.4f}")
        print(f"  Écart-type: {np.std(data):.4f}")
        print(f"  Min: {np.min(data):.4f}")
        print(f"  Max: {np.max(data):.4f}")