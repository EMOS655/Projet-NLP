import re
import logging
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
import spacy
from tqdm import tqdm

# --- CONFIGURATION DES CHEMINS ---
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    import config
    from src.utils import save_results, setup_logging, Timer
    logger = setup_logging()
except ImportError as e:
    print(f"Erreur d'importation : {e}. Vérifiez la structure du projet.")
    sys.exit(1)

class PDFProcessor:
    """Processeur optimisé pour l'audit sémantique des Lois de Finances et SND30."""
    
    def __init__(self):
        logger.info("Initialisation du PDFProcessor avec SpaCy")
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            # Ajout de stopwords spécifiques au corpus budgétaire camerounais
            self.budget_stopwords = {
                "article", "alinéa", "loi", "finances", "budget", "chapitre", 
                "paragraphe", "dispositions", "exercice", "montant", "publique",
                "dépenses", "recettes", "visa", "conformément", "vigueur"
            }
        except OSError:
            logger.error(f"Modèle {config.SPACY_MODEL} introuvable.")
            raise

    def is_table_noise(self, line: str) -> bool:
        """Détecte si une ligne est du bruit issu d'un tableau (chiffres dominants)."""
        clean_line = line.strip()
        if not clean_line:
            return True
        digits = sum(c.isdigit() for c in clean_line)
        # Si plus de 40% de la ligne sont des chiffres, c'est probablement une dotation
        return (digits / len(clean_line)) > 0.4

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extraie le texte en filtrant le bruit des tableaux budgétaires."""
        logger.info(f"Extraction sémantique du PDF: {pdf_path.name}")
        text_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in tqdm(pdf.pages, desc=f"Extraction {pdf_path.name}"):
                    page_text = page.extract_text()
                    if page_text:
                        # Filtrage ligne par ligne pour éliminer les tableaux de chiffres purs
                        lines = [line for line in page_text.split('\n') if not self.is_table_noise(line)]
                        text_content.append("\n".join(lines))
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Erreur extraction {pdf_path.name}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Nettoyage approfondi pour stabiliser les plongements vectoriels."""
        # 1. Normalisation Unicode (enlève les accents si config.STRIP_ACCENTS)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Normalisation des apostrophes (Crucial pour 'droit d'accise' vs 'droit d’accise')
        text = re.sub(r"[’'‘`]", "'", text)
        
        # 3. Suppression des césures de mots en fin de ligne
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 4. Nettoyage des espaces et sauts de page
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+ sur \d+', '', text)
        
        if config.LOWERCASE:
            text = text.lower()
            
        return text.strip()

    def get_semantic_tokens(self, text: str) -> str:
        """Prépare le texte pour CamemBERT en filtrant le bruit administratif."""
        doc = self.nlp(text)
        # On garde les lemmes, on exclut la ponctuation, les chiffres et les mots vides budgétaires
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.like_num
            and token.lemma_.lower() not in self.budget_stopwords
            and len(token.text) > 2
        ]
        return " ".join(tokens)

    def extract_articles(self, text: str) -> List[Dict]:
        """
        Découpe le texte en articles pour la classification Zero-shot[cite: 9].
        Capture le numéro, le titre (souvent porteur de sens) et le contenu.
        """
        articles = []
        # Pattern robuste : Article + Numéro + Titre optionnel jusqu'au prochain article
        pattern = r'(?:ARTICLE|Art\.)\s*(\d+)\s*[:\.-]?\s*([^\n]*?)\.\s*(.*?)(?=(?:ARTICLE|Art\.)\s*\d+|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            num = match.group(1)
            titre = match.group(2).strip()
            contenu = match.group(3).strip()
            
            # On crée une version 'nettoyée' uniquement pour le calcul de similarité
            full_text = f"{titre} {contenu}"
            semantic_text = self.get_semantic_tokens(full_text)
            
            if len(semantic_text.split()) >= config.MIN_SENTENCE_LENGTH:
                articles.append({
                    "numero": num,
                    "titre": titre,
                    "contenu_brut": contenu[:500] + "...", # Pour affichage UI
                    "semantic_input": semantic_text, # Input pour CamemBERT
                    "metadata": {"length": len(contenu)}
                })
                
        logger.info(f"{len(articles)} articles sémantiques extraits")
        return articles

    def process_document(self, pdf_path: Path, is_snd30: bool = False) -> Dict:
        """Pipeline complet de traitement d'un document budgétaire."""
        with Timer(f"Processing {pdf_path.name}"):
            raw_text = self.extract_text_from_pdf(pdf_path)
            clean_text = self.clean_text(raw_text)
            
            result = {
                'filename': pdf_path.name,
                'statistics': {'raw_length': len(raw_text)}
            }
            
            if is_snd30:
                # Pour la SND30, on segmente par paragraphes thématiques
                paragraphs = [p.strip() for p in clean_text.split('.') if len(p.split()) > 10]
                result['segments'] = [{"text": p, "semantic": self.get_semantic_tokens(p)} for p in paragraphs]
            else:
                # Pour les Lois de Finances, on travaille par article 
                result['articles'] = self.extract_articles(clean_text)
                result['statistics']['num_articles'] = len(result['articles'])
            
            return result

def preprocess_all_documents():
    """Point d'entrée principal pour préparer les données de l'audit[cite: 16]."""
    processor = PDFProcessor()
    final_data = {}
    
    docs = {
        "2024": config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2024],
        "2025": config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2025],
        "SND30": config.RAW_DATA_DIR / config.DOCUMENT_NAMES['SND30']
    }
    
    for key, path in docs.items():
        if path.exists():
            is_snd = (key == "SND30")
            final_data[key] = processor.process_document(path, is_snd30=is_snd)
            save_results(final_data[key], f"processed_{key}", format='json')
        else:
            logger.error(f"Fichier manquant: {path}")

    save_results(final_data, "all_processed_data", format='json')
    logger.info("Prétraitement terminé avec succès.")

if __name__ == "__main__":
    preprocess_all_documents()