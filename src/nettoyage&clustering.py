import os
import sys

# --- CONFIGURATION DES CHEMINS ---
# On récupère le chemin absolu de la racine (Projet-NLP)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# On ajoute la racine ET le dossier src au chemin de recherche de Python
if root_path not in sys.path:
    sys.path.insert(0, root_path)
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORTS ---
# Maintenant, config est importé depuis la racine
# Et les autres depuis le dossier src
try:
    import config
    from extrator import PDFExtractor
    from semantic_engine import SemanticAudit
    from financial_correlation import FinancialCorrelation
    
    RESULTS_DIR = config.RESULTS_DIR
except ImportError as e:
    print(f"[!] Erreur d'importation : {e}")
    print("Vérifiez que vos fichiers sont bien nommés (ex: extrator.py et non extractor.py)")
    sys.exit(1)

def run_full_audit():
    print("\n" + "="*60)
    print("   LANCEMENT GLOBAL DE L'AUDIT SÉMANTIQUE ISSEA")
    print("="*60)

    # 1. INITIALISATION
    # Les classes utilisent les variables de config.py en interne
    extractor = PDFExtractor()
    audit_engine = SemanticAudit()
    finance_engine = FinancialCorrelation()

    # 2. ÉTAPE D'EXTRACTION (Avec le nouveau filtre anti-bruit)
    print("\n[1/3] Extraction et nettoyage des données PDF...")
    df_raw = extractor.process_all_pdfs()

    if df_raw is None or df_raw.empty:
        print("[!] Erreur : Aucune donnée extraite. Vérifiez 'data/raw/'")
        return

    # 3. ÉTAPE DE CLASSIFICATION & CLUSTERING
    print("\n[2/3] Classification SND30 et Clustering K-Means...")
    df_classified = audit_engine.classify_and_cluster(df_raw, n_clusters=5)

    # Sauvegarde du fichier pivot
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    save_path = os.path.join(RESULTS_DIR, "audit_results.csv")
    df_classified.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[+] Résultats sauvegardés : {save_path}")

    # 4. ÉTAPE D'ANALYSE FINANCIÈRE
    print("\n[3/3] Calcul des corrélations financières réalistes...")
    finance_engine.run_analysis()

    print("\n" + "="*60)
    print("   PIPELINE TERMINÉ AVEC SUCCÈS")
    print("   Visualisation : streamlit run dashboard/app.py")
    print("="*60)

if __name__ == "__main__":
    run_full_audit()