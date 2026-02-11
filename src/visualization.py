import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

# Ajout du chemin projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import config

def generate_comparison_plot():
    # 1. Charger les résultats
    results_path = config.RESULTS_DIR / "comparison_piliers_2024_2025.json"
    if not results_path.exists():
        print("Erreur : Fichier de comparaison introuvable.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Préparer les données pour Seaborn
    plot_data = []
    for year, dist in data.items():
        for pillar, count in dist.items():
            plot_data.append({
                "Année": year,
                "Pilier SND30": pillar,
                "Nombre d'articles": count
            })
    
    df = pd.DataFrame(plot_data)

    # 3. Création du graphique
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Palette de couleurs pro
    palette = {"2024": "#3498db", "2025": "#e74c3c"}
    
    ax = sns.barplot(
        data=df, 
        x="Pilier SND30", 
        y="Nombre d'articles", 
        hue="Année",
        palette=palette
    )

    # Personnalisation
    plt.title("Comparaison de l'alignement budgétaire : 2024 vs 2025", fontsize=15, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nombre d'articles classifiés")
    plt.xlabel("")
    
    # Ajouter les chiffres au-dessus des barres
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    plt.tight_layout()

    # 4. Sauvegarde
    output_path = config.RESULTS_DIR / "comparaison_piliers_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Graphique sauvegardé dans : {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot()