"""
Script de diagnostic pour analyser le contenu des articles budgétaires
et comprendre pourquoi ils sont tous classés comme "Capital humain"
"""

import json
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent))

try:
    from config import PROCESSED_DATA_DIR, SND30_PILLARS
except:
    PROCESSED_DATA_DIR = Path("data/processed")
    SND30_PILLARS = ["Transformation structurelle", "Capital humain", "Gouvernance", "Développement régional", "Autre"]

def analyze_budget_data(year=2024):
    """Analyse le contenu des données budgétaires"""
    
    data_file = PROCESSED_DATA_DIR / f"processed_{year}.json"
    
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYSE DES DONNÉES BUDGÉTAIRES {year}")
    print(f"{'='*80}\n")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('articles', [])
    
    # Vérifier le type de données
    if isinstance(items, dict):
        # Si c'est un dict, convertir en liste
        items_list = list(items.values())
        print(f"⚠️  Structure: Dictionnaire (converti en liste)")
    elif isinstance(items, list):
        items_list = items
        print(f"✅ Structure: Liste")
    else:
        print(f"❌ Structure inconnue: {type(items)}")
        return
    
    print(f"📊 Nombre total d'articles: {len(items_list)}")
    
    if items_list:
        print(f"📄 Type de données: {type(items_list[0])}\n")
    else:
        print("📄 Type de données: N/A (pas d'articles)\n")
        return
    
    # Analyser les 5 premiers articles
    print(f"{'='*80}")
    print("EXEMPLES D'ARTICLES (5 premiers)")
    print(f"{'='*80}\n")
    
    for i, item in enumerate(items_list[:5], 1):
        print(f"\n--- Article {i} ---")
        
        if isinstance(item, str):
            print(f"Type: String")
            print(f"Contenu: {item[:200]}...")
            print(f"Longueur: {len(item)} caractères")
        
        elif isinstance(item, dict):
            print(f"Type: Dictionnaire")
            print(f"Clés: {list(item.keys())}")
            
            if 'numero' in item:
                print(f"Numéro: {item.get('numero')}")
            if 'titre' in item:
                print(f"Titre: {item.get('titre', '')[:100]}")
            if 'contenu' in item:
                contenu = item.get('contenu', '')
                print(f"Contenu: {contenu[:200]}...")
                print(f"Longueur contenu: {len(contenu)} caractères")
        else:
            print(f"Type inconnu: {type(item)}")
    
    # Statistiques sur les longueurs
    print(f"\n{'='*80}")
    print("STATISTIQUES")
    print(f"{'='*80}\n")
    
    lengths = []
    for item in items_list:
        if isinstance(item, str):
            lengths.append(len(item))
        elif isinstance(item, dict):
            text = f"{item.get('titre', '')} {item.get('contenu', '')}"
            lengths.append(len(text))
    
    if lengths:
        print(f"Longueur moyenne: {sum(lengths)/len(lengths):.0f} caractères")
        print(f"Longueur min: {min(lengths)} caractères")
        print(f"Longueur max: {max(lengths)} caractères")
        
        # Articles courts vs longs
        short = sum(1 for l in lengths if l < 100)
        medium = sum(1 for l in lengths if 100 <= l < 500)
        long = sum(1 for l in lengths if l >= 500)
        
        print(f"\n📏 Distribution par longueur:")
        print(f"  • Courts (< 100 chars): {short} ({short/len(lengths)*100:.1f}%)")
        print(f"  • Moyens (100-500): {medium} ({medium/len(lengths)*100:.1f}%)")
        print(f"  • Longs (> 500): {long} ({long/len(lengths)*100:.1f}%)")
    
    # Recherche de mots-clés par pilier
    print(f"\n{'='*80}")
    print("DÉTECTION DE MOTS-CLÉS PAR PILIER")
    print(f"{'='*80}\n")
    
    keywords = {
        "Transformation structurelle": ["infrastructure", "route", "électrification", "industrie", "agriculture", "mine", "énergie"],
        "Capital humain": ["éducation", "santé", "école", "hôpital", "formation", "social", "emploi"],
        "Gouvernance": ["justice", "sécurité", "police", "armée", "administration", "corruption", "transparence"],
        "Développement régional": ["région", "rural", "local", "territoire", "commune", "développement local"],
        "Autre": ["budget", "crédit", "impôt", "dette", "fiscal", "trésorerie"]
    }
    
    for pillar, kws in keywords.items():
        count = 0
        for item in items_list:
            text = ""
            if isinstance(item, str):
                text = item.lower()
            elif isinstance(item, dict):
                text = f"{item.get('titre', '')} {item.get('contenu', '')}".lower()
            
            if any(kw in text for kw in kws):
                count += 1
        
        print(f"  • {pillar}: {count} articles ({count/len(items_list)*100:.1f}%)")


if __name__ == "__main__":
    print("\n🔍 DIAGNOSTIC DES DONNÉES BUDGÉTAIRES\n")
    
    # Analyser 2024 et 2025
    for year in [2024, 2025]:
        analyze_budget_data(year)
    
    print(f"\n{'='*80}")
    print("RECOMMANDATIONS")
    print(f"{'='*80}\n")
    print("""
Si la plupart des articles sont très courts (< 100 caractères), 
le problème est que CamemBERT n'a pas assez d'informations pour classifier correctement.

SOLUTIONS:
1. Vérifier que les fichiers processed_YYYY.json contiennent le texte complet des articles
2. Si les articles sont vraiment courts, enrichir avec des métadonnées (numéro, section, etc.)
3. Fine-tuner CamemBERT avec des exemples labellisés manuellement
4. Utiliser les sections budgétaires au lieu des articles individuels
    """)