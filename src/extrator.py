import os
import sys
import pdfplumber
import pandas as pd
import re

# --- GESTION DES CHEMINS ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from src.config import RAW_DATA_DIR
except ImportError:
    import config
    RAW_DATA_DIR = config.RAW_DATA_DIR

class PDFExtractor:
    def __init__(self):
        self.raw_path = RAW_DATA_DIR
        # Liste noire étendue pour capturer le bruit administratif et juridique
        self.blacklist = [
            r"REPUBLIQUE DU CAMEROUN", r"PAIX\s?-\s?TRAVAIL\s?-\s?PATRIE",
            r"PRESIDENCE DE LA REPUBLIQUE", r"MINISTERE DE", r"LOI DE FINANCES",
            r"LE PARLEMENT A DELIBERE", r"ARTICLE\s?\d+", r"EXERCICE\s?\d{4}",
            r"CHAPITRE", r"SECTION", r"PARAGRAPHE", r"LIVRE", r"TITRE\s?\w+",
            r"SOMMAIRE", r"PAGE\s?\d+", r"DUREE DE", r"COMPTER DU",
            r"NOMENCLATURE", r"CODE\s?\d+", r"CLASSEMENT", r"TOTAL",
            r"PROJET DE LOI", r"REPUBLIQUE DU CAMl;ROUN" # Correction OCR
        ]

    def is_valid_project(self, line):
        """
        Filtre avancé pour nettoyer le Cluster 0 (Bruit).
        """
        line = line.strip()
        
        # 1. Filtre de longueur : Un vrai libellé de projet est descriptif
        if len(line) < 45: 
            return False
            
        # 2. Éliminer le bruit juridique (ex: quinquies, sexies, i., ii., etc.)
        # Supprime les lignes contenant des énumérations latines ou chiffres romains de liste
        legal_noise = r'\b(v|i+|x+)\b\.|\b\w+(ies|ies,)\b'
        if re.search(legal_noise, line, re.IGNORECASE):
            return False

        # 3. Filtre de densité de symboles (Cible les lignes de pointillés .......)
        # On compte les caractères qui ne sont ni des lettres, ni des chiffres, ni des espaces
        special_chars = sum(1 for c in line if not c.isalnum() and not c.isspace())
        if (special_chars / len(line)) > 0.15: # Si plus de 15% de symboles, c'est du bruit
            return False

        # 4. Filtre les majuscules abusives (Titres de pages ou en-têtes)
        if line.isupper() and len(line) < 120:
            return False

        # 5. Vérification de la Blacklist Regex
        for pattern in self.blacklist:
            if re.search(pattern, line, re.IGNORECASE):
                return False

        return True

    def extract_text_from_pdf(self, file_path):
        """Extraction brute avec pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            print(f"   [!] Erreur lors de la lecture de {file_path}: {e}")
        return text

    def process_all_pdfs(self):
        """Parcourt, extrait et nettoie les données."""
        all_data = []
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)
            
        files = [f for f in os.listdir(self.raw_path) if f.endswith('.pdf')]
        
        if not files:
            print(f"   [!] Aucun fichier PDF trouvé dans {self.raw_path}")
            return pd.DataFrame()

        for file_name in files:
            print(f"   [>] Nettoyage intensif de : {file_name}...")
            file_path = os.path.join(self.raw_path, file_name)
            
            # Détection de l'année
            annee_match = re.search(r'\d{4}', file_name)
            annee = int(annee_match.group()) if annee_match else 2024
            
            raw_text = self.extract_text_from_pdf(file_path)
            lines = raw_text.split('\n')
            
            # Application du nouveau filtre
            projets_valides = [l.strip() for l in lines if self.is_valid_project(l)]
            
            for p in projets_valides:
                all_data.append({
                    'exercice': annee,
                    'libelle_projet': p
                })
        
        return pd.DataFrame(all_data)

if __name__ == "__main__":
    extractor = PDFExtractor()
    df = extractor.process_all_pdfs()
    print(f"\n[+] Extraction terminée : {len(df)} lignes valides conservées.")
    if not df.empty:
        print("--- Aperçu des 5 premières lignes ---")
        print(df['libelle_projet'].head())