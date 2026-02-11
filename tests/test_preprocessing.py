import sys
import os
from pathlib import Path

# --- CONFIGURATION DU CHEMIN D'ACCÈS ---
# On remonte d'un niveau pour atteindre la racine du projet (Projet-NLP)
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    # On importe depuis src car votre exécution précédente a montré 
    # que le fichier est dans le dossier src/
    from src.preprocessing import PDFProcessor
    import config
    print("✅ Importation réussie : src.preprocessing et config trouvés.")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print(f"Détail : Python cherche dans {sys.path[0]}")
    print("Vérifiez que vous avez bien un fichier src/preprocessing.py")
    sys.exit(1)

def test_preprocessing():
    """Fonction de test pour valider le pipeline de nettoyage et d'extraction"""
    print("\n" + "="*50)
    print("DÉBUT DU TEST DE COMPATIBILITÉ PREPROCESSING")
    print("="*50)
    
    processor = PDFProcessor()
    
    # On définit le chemin du fichier de test (Loi de finances 2024)
    # Assurez-vous que le nom dans config.DOCUMENT_NAMES[2024] est correct
    pdf_filename = config.DOCUMENT_NAMES.get(2024, "Loi_de_Finances_2024.pdf")
    test_pdf = config.RAW_DATA_DIR / pdf_filename
    
    if not test_pdf.exists():
        print(f"❌ Erreur : Le fichier {test_pdf} est introuvable.")
        print(f"Vérifiez qu'il est bien dans : {config.RAW_DATA_DIR}")
        return

    try:
        # 1. Test du traitement complet
        print(f"🔄 Test du traitement sur : {pdf_filename}...")
        data = processor.process_document(test_pdf, extract_articles=True)
        
        # 2. Vérification de la structure (CRUCIAL pour le classificateur)
        print("\n--- Analyse des résultats ---")
        
        # Vérification des articles
        articles = data.get('articles', [])
        if isinstance(articles, list):
            print(f"✅ SUCCESS: 'articles' est une LISTE ({len(articles)} extraits).")
            if len(articles) > 0:
                # Vérification du format du premier article
                first = articles[0]
                if all(k in first for k in ['numero', 'contenu']):
                    print(f"✅ SUCCESS: Structure d'article valide (Art. {first['numero']})")
                else:
                    print(f"❌ ERROR: Clés 'numero' ou 'contenu' manquantes dans l'article.")
        else:
            print("❌ ERROR: 'articles' n'est pas une liste. Le classificateur va échouer.")

        # 3. Test de la normalisation du texte
        test_text = "L’investissement pour l'économie" # Notez l'apostrophe courbe
        cleaned = processor.clean_text(test_text)
        if "’" not in cleaned:
            print("✅ SUCCESS: Apostrophes courbes normalisées en ' (crucial pour les mots-clés).")
        else:
            print("⚠️ WARNING: Les apostrophes courbes persistent dans le texte.")

        # 4. Vérification des statistiques
        if 'statistics' in data and data['statistics'].get('num_articles', 0) > 0:
            print(f"✅ SUCCESS: Statistiques générées avec succès.")

        print("\n" + "="*50)
        print("FÉLICITATIONS : Le preprocessing est prêt pour la classification !")
        print("="*50)

    except Exception as e:
        print(f"❌ Une erreur est survenue pendant le test : {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing()