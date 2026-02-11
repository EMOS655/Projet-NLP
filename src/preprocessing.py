import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
import spacy
from tqdm import tqdm
#import config
#from src.utils import save_results, setup_logging, Timer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import save_results, setup_logging, Timer
# Setup logging
logger = setup_logging()


class PDFProcessor:
    """Classe pour traiter les fichiers PDF"""
    
    def __init__(self):
        """Initialise le processeur PDF"""
        logger.info("Initialisation du PDFProcessor")
        
        # Charger le modèle SpaCy pour le français
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            logger.info(f"Modèle SpaCy chargé: {config.SPACY_MODEL}")
        except OSError:
            logger.error(f"Modèle SpaCy non trouvé: {config.SPACY_MODEL}")
            logger.info("Installez-le avec: python -m spacy download fr_core_news_md")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extrait le texte d'un fichier PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Texte extrait
        """
        logger.info(f"Extraction du PDF: {pdf_path.name}")
        
        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Nombre de pages: {len(pdf.pages)}")
                
                for i, page in enumerate(tqdm(pdf.pages, desc=f"Extraction {pdf_path.name}")):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
            logger.info(f"Extraction terminée: {len(text)} caractères")
            return text
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte extrait
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        logger.info("Nettoyage du texte")
        
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 sauts de ligne consécutifs
        text = re.sub(r' {2,}', ' ', text)  # Max 1 espace
        text = re.sub(r'\t+', ' ', text)  # Remplacer tabs par espaces
        
        # Supprimer les numéros de page isolés
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Supprimer les tirets de césure en fin de ligne
        text = re.sub(r'-\n', '', text)
        
        # Normaliser les apostrophes
        text = text.replace("'", "'").replace("`", "'")
        
        # Supprimer les espaces avant ponctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        if config.LOWERCASE:
            text = text.lower()
        
        logger.info(f"Nettoyage terminé: {len(text)} caractères")
        return text.strip()
    
    def segment_into_sentences(self, text: str) -> List[str]:
        """
        Segmente le texte en phrases avec SpaCy
        
        Args:
            text: Texte à segmenter
            
        Returns:
            Liste de phrases
        """
        logger.info("Segmentation en phrases")
        
        # Limiter la taille du texte pour SpaCy (max 1M caractères par chunk)
        max_length = 1000000
        sentences = []
        
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            doc = self.nlp(chunk)
            
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                
                # Filtrer les phrases trop courtes ou trop longues
                word_count = len(sentence_text.split())
                
                if config.MIN_SENTENCE_LENGTH <= word_count <= config.MAX_SENTENCE_LENGTH:
                    sentences.append(sentence_text)
        
        logger.info(f"Segmentation terminée: {len(sentences)} phrases")
        return sentences
    
    def segment_into_paragraphs(self, text: str) -> List[str]:
        """
        Segmente le texte en paragraphes
        
        Args:
            text: Texte à segmenter
            
        Returns:
            Liste de paragraphes
        """
        logger.info("Segmentation en paragraphes")
        
        # Diviser par doubles sauts de ligne
        paragraphs = text.split('\n\n')
        
        # Nettoyer et filtrer
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraphs = [p for p in paragraphs if len(p.split()) >= config.MIN_SENTENCE_LENGTH]
        
        logger.info(f"Segmentation terminée: {len(paragraphs)} paragraphes")
        return paragraphs
    
    def extract_articles(self, text: str) -> Dict[str, str]:
        """
        Extrait les articles de loi du texte
        
        Args:
            text: Texte de la loi
            
        Returns:
            Dictionnaire {numéro_article: contenu}
        """
        logger.info("Extraction des articles")
        
        articles = {}
        
        # Pattern pour détecter les articles
        # Ex: "Article 1", "Art. 2", "ARTICLE 3"
        pattern = r'(?:ARTICLE|Article|ART\.|Art\.)\s*(\d+)[:\.]?\s*[–-]?\s*(.*?)(?=(?:ARTICLE|Article|ART\.|Art\.)\s*\d+|$)'
        
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            article_num = match.group(1)
            article_text = match.group(2).strip()
            
            if article_text:  # Ne garder que les articles avec contenu
                articles[f"Article {article_num}"] = article_text
        
        logger.info(f"Extraction terminée: {len(articles)} articles trouvés")
        return articles
    
    def process_document(self, pdf_path: Path, extract_articles: bool = False) -> Dict:
        """
        Traite un document PDF complet
        
        Args:
            pdf_path: Chemin vers le PDF
            extract_articles: Si True, extrait aussi les articles de loi
            
        Returns:
            Dictionnaire avec texte brut, nettoyé, phrases, paragraphes, articles
        """
        logger.info(f"Traitement complet du document: {pdf_path.name}")
        
        with Timer(f"Traitement {pdf_path.name}"):
            # Extraction
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Nettoyage
            clean_text = self.clean_text(raw_text)
            
            # Segmentation
            sentences = self.segment_into_sentences(clean_text)
            paragraphs = self.segment_into_paragraphs(clean_text)
            
            result = {
                'filename': pdf_path.name,
                'raw_text': raw_text,
                'clean_text': clean_text,
                'sentences': sentences,
                'paragraphs': paragraphs,
                'statistics': {
                    'raw_length': len(raw_text),
                    'clean_length': len(clean_text),
                    'num_sentences': len(sentences),
                    'num_paragraphs': len(paragraphs)
                }
            }
            
            # Extraction des articles si demandé
            if extract_articles:
                articles = self.extract_articles(clean_text)
                result['articles'] = articles
                result['statistics']['num_articles'] = len(articles)
            
            logger.info(f"Traitement terminé: {result['statistics']}")
            
            return result


def preprocess_all_documents(save_outputs: bool = True) -> Dict[str, Dict]:
    """
    Prétraite tous les documents PDF du projet
    
    Args:
        save_outputs: Si True, sauvegarde les résultats
        
    Returns:
        Dictionnaire {année: données_traitées}
    """
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DU PREPROCESSING DE TOUS LES DOCUMENTS")
    logger.info("=" * 80)
    
    processor = PDFProcessor()
    results = {}
    
    # Documents à traiter
    documents_to_process = {
        2024: config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2024],
        2025: config.RAW_DATA_DIR / config.DOCUMENT_NAMES[2025],
        'SND30': config.RAW_DATA_DIR / config.DOCUMENT_NAMES['SND30']
    }
    
    for key, pdf_path in documents_to_process.items():
        if not pdf_path.exists():
            logger.warning(f"Fichier manquant: {pdf_path}")
            continue
        
        try:
            # Traiter le document (extraire articles pour les lois de finances)
            extract_articles = key != 'SND30'
            result = processor.process_document(pdf_path, extract_articles=extract_articles)
            results[str(key)] = result
            
            # Sauvegarder individuellement
            if save_outputs:
                output_file = config.PROCESSED_DATA_DIR / f"processed_{key}.json"
                save_results(result, f"processed_{key}", format='json')
                logger.info(f"Sauvegarde: {output_file}")
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {pdf_path}: {e}")
            continue
    
    # Sauvegarder tous les résultats ensemble
    if save_outputs and results:
        save_results(results, "all_processed_documents", format='json')
    
    logger.info("=" * 80)
    logger.info("PREPROCESSING TERMINÉ")
    logger.info(f"Documents traités: {len(results)}")
    logger.info("=" * 80)
    
    return results


def get_processed_texts(year: int = None) -> Dict:
    """
    Charge les textes prétraités
    
    Args:
        year: Année spécifique ou None pour tous
        
    Returns:
        Textes prétraités
    """
    if year:
        filepath = config.PROCESSED_DATA_DIR / f"processed_{year}.json"
        if filepath.exists():
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.error(f"Fichier non trouvé: {filepath}")
            return {}
    else:
        filepath = config.PROCESSED_DATA_DIR / "all_processed_documents.json"
        if filepath.exists():
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.error(f"Fichier non trouvé: {filepath}")
            return {}


if __name__ == "__main__":
    # Test du preprocessing
    print("Démarrage du preprocessing...")
    results = preprocess_all_documents(save_outputs=True)
    
    # Afficher les statistiques
    print("\n" + "=" * 80)
    print("RÉSUMÉ DU PREPROCESSING")
    print("=" * 80)
    
    for key, data in results.items():
        print(f"\n{key}:")
        print(f"  - Caractères (brut): {data['statistics']['raw_length']:,}")
        print(f"  - Caractères (nettoyé): {data['statistics']['clean_length']:,}")
        print(f"  - Phrases: {data['statistics']['num_sentences']:,}")
        print(f"  - Paragraphes: {data['statistics']['num_paragraphs']:,}")
        if 'num_articles' in data['statistics']:
            print(f"  - Articles: {data['statistics']['num_articles']:,}")