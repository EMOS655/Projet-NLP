import re
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# --- RÉPARATION DES CHEMINS ---
# On ajoute la racine du projet au chemin de recherche de Python
# pour pouvoir importer 'config' et 'src' correctement.
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pdfplumber
import spacy
from tqdm import tqdm

# Import des modules locaux
try:
    import config
    from src.utils import save_results, setup_logging, Timer
    logger = setup_logging()
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print(f"Vérifiez que config.py est à la racine : {root_path}")
    sys.exit(1)

class PDFProcessor:
    """Classe pour traiter les fichiers PDF et les préparer pour l'audit sémantique"""
    
    def __init__(self):
        """Initialise le processeur PDF avec SpaCy"""
        logger.info("Initialisation du PDFProcessor")
        try:
            # On utilise le modèle défini dans config.py
            self.nlp = spacy.load(config.SPACY_MODEL)
            logger.info(f"Modèle SpaCy chargé: {config.SPACY_MODEL}")
        except OSError:
            logger.error(f"Modèle SpaCy non trouvé: {config.SPACY_MODEL}")
            logger.info("Installez-le avec: python -m spacy download fr_core_news_md")
            raise

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extrait le texte brut d'un fichier PDF"""
        logger.info(f"Extraction du PDF: {pdf_path.name}")
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in tqdm(pdf.pages, desc=f"Extraction {pdf_path.name}"):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Nettoie le texte (Apostrophes, espaces, sauts de ligne)"""
        logger.info("Nettoyage du texte")
        
        # 1. Normalisation des apostrophes (Crucial pour la détection de mots-clés)
        text = re.sub(r"[’'‘`]", "'", text)
        
        # 2. Nettoyage des espaces et sauts de ligne
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # 3. Supprimer les numéros de page isolés et les césures
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'-\n', '', text)
        
        # 4. Supprimer les espaces avant ponctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        if config.LOWERCASE:
            text = text.lower()
        
        return text.strip()

    def segment_into_sentences(self, text: str) -> List[str]:
        """Segmente le texte en phrases avec SpaCy"""
        logger.info("Segmentation en phrases")
        max_length = 1000000
        sentences = []
        
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            doc = self.nlp(chunk)
            for sent in doc.sents:
                s_text = sent.text.strip()
                word_count = len(s_text.split())
                if config.MIN_SENTENCE_LENGTH <= word_count <= config.MAX_SENTENCE_LENGTH:
                    sentences.append(s_text)
        return sentences

    def extract_articles(self, text: str) -> List[Dict]:
        """
        Extrait les articles sous forme de LISTE de dictionnaires.
        Format attendu par le classificateur hybride.
        """
        logger.info("Extraction des articles")
        articles = []
        
        # Pattern pour capturer "Article X", "Art. X" et le contenu jusqu'au prochain article
        pattern = r'(?:ARTICLE|Article|ART\.|Art\.)\s*(\d+)[:\.]?\s*[–-]?\s*(.*?)(?=(?:ARTICLE|Article|ART\.|Art\.)\s*\d+|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            articles.append({
                "numero": match.group(1),
                "titre": f"Article {match.group(1)}",
                "contenu": match.group(2).strip()
            })
            
        logger.info(f"{len(articles)} articles extraits")
        return articles

    def process_document(self, pdf_path: Path, extract_articles: bool = False) -> Dict:
        """Traite un document complet"""
        logger.info(f"Début du traitement de : {pdf_path.name}")
        
        with Timer(f"Traitement {pdf_path.name}"):
            raw_text = self.extract_text_from_pdf(pdf_path)
            clean_text = self.clean_text(raw_text)
            
            sentences = self.segment_into_sentences(clean_text)
            
            result = {
                'filename': pdf_path.name,
                'raw_text': raw_text,
                'clean_text': clean_text,
                'sentences': sentences,
                'statistics': {
                    'raw_length': len(raw_text),
                    'clean_length': len(clean_text),
                    'num_sentences': len(sentences)
                }
            }
            
            if extract_articles:
                arts = self.extract_articles(clean_text)
                result['articles'] = arts # Liste de dictionnaires
                result['statistics']['num_articles'] = len(arts)
            
            return result

def preprocess_all_documents(save_outputs: bool = True):
    """Prétraite tous les documents définis dans la config"""
    logger.info("DÉMARRAGE DU PREPROCESSING GLOBAL")
    
    processor = PDFProcessor()
    results = {}
    
    # Documents à traiter (clés 2024, 2025, 'SND30')
    documents_to_process = {
        2024: config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2024],
        2025: config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2025],
        'SND30': config.RAW_DATA_DIR / config.DOCUMENT_NAMES['SND30']
    }
    
    for key, pdf_path in documents_to_process.items():
        if not pdf_path.exists():
            logger.warning(f"Fichier manquant : {pdf_path}")
            continue
        
        try:
            # On extrait les articles pour les lois de finances, pas pour la SND30
            extract_arts = (key != 'SND30')
            result = processor.process_document(pdf_path, extract_articles=extract_arts)
            results[str(key)] = result
            
            if save_outputs:
                save_results(result, f"processed_{key}", format='json')
        except Exception as e:
            logger.error(f"Erreur sur {pdf_path.name}: {e}")
            
    if save_outputs and results:
        save_results(results, "all_processed_documents", format='json')
    
    logger.info("PREPROCESSING TERMINÉ")
    return results

if __name__ == "__main__":
    preprocess_all_documents()