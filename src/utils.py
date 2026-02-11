import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd
import config


def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Configure le système de logging pour le projet
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger configuré
    """
    if log_level is None:
        log_level = config.LOGGING_LEVEL
    
    # Créer le dossier logs s'il n'existe pas
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('NLP_Project')
    logger.info("Logging system initialized")
    
    return logger


def save_results(data: Any, filename: str, format: str = 'json') -> Path:
    """
    Sauvegarde des résultats dans différents formats
    
    Args:
        data: Données à sauvegarder
        filename: Nom du fichier (sans extension)
        format: Format de sauvegarde ('json', 'csv', 'pickle', 'txt')
        
    Returns:
        Path: Chemin du fichier sauvegardé
    """
    # Déterminer le chemin de sauvegarde
    if format == 'json':
        filepath = config.RESULTS_DIR / f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    elif format == 'csv':
        filepath = config.RESULTS_DIR / f"{filename}.csv"
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False, encoding='utf-8')
        else:
            pd.DataFrame(data).to_csv(filepath, index=False, encoding='utf-8')
    
    elif format == 'pickle':
        filepath = config.RESULTS_DIR / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    elif format == 'txt':
        filepath = config.RESULTS_DIR / f"{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    logging.info(f"Résultats sauvegardés: {filepath}")
    return filepath


def load_results(filename: str, format: str = 'json') -> Any:
    """
    Charge des résultats depuis un fichier
    
    Args:
        filename: Nom du fichier (avec ou sans extension)
        format: Format du fichier ('json', 'csv', 'pickle', 'txt')
        
    Returns:
        Données chargées
    """
    # Ajouter l'extension si nécessaire
    if not filename.endswith(('.json', '.csv', '.pkl', '.txt')):
        if format == 'json':
            filename += '.json'
        elif format == 'csv':
            filename += '.csv'
        elif format == 'pickle':
            filename += '.pkl'
        elif format == 'txt':
            filename += '.txt'
    
    filepath = config.RESULTS_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    if format == 'json' or filename.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    elif format == 'csv' or filename.endswith('.csv'):
        data = pd.read_csv(filepath, encoding='utf-8')
    
    elif format == 'pickle' or filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
    elif format == 'txt' or filename.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
    
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    logging.info(f"Résultats chargés: {filepath}")
    return data


def create_directories():
    """
    Crée tous les dossiers nécessaires pour le projet
    """
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.RESULTS_DIR,
        config.PRETRAINED_DIR,
        config.FINETUNED_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.REPORTS_DIR,
        config.LOG_FILE.parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Dossier créé/vérifié: {directory}")


def get_pdf_files() -> Dict[str, Path]:
    """
    Récupère les chemins des fichiers PDF
    
    Returns:
        Dict: Dictionnaire {année/type: chemin_fichier}
    """
    pdf_files = {}
    
    for key, filename in config.DOCUMENT_NAMES.items():
        filepath = config.RAW_DATA_DIR / filename
        if filepath.exists():
            pdf_files[key] = filepath
            logging.info(f"PDF trouvé: {key} -> {filepath}")
        else:
            logging.warning(f"PDF manquant: {key} -> {filepath}")
    
    return pdf_files


def format_number(number: float, decimals: int = 2) -> str:
    """
    Formate un nombre pour l'affichage
    
    Args:
        number: Nombre à formater
        decimals: Nombre de décimales
        
    Returns:
        Nombre formaté en string
    """
    return f"{number:,.{decimals}f}".replace(',', ' ')


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calcule le pourcentage de changement entre deux valeurs
    
    Args:
        old_value: Ancienne valeur
        new_value: Nouvelle valeur
        
    Returns:
        Pourcentage de changement
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else 0
    
    return ((new_value - old_value) / old_value) * 100


def text_statistics(text: str) -> Dict[str, int]:
    """
    Calcule des statistiques basiques sur un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Dictionnaire de statistiques
    """
    words = text.split()
    sentences = text.split('.')
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0
    }


def progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
    """
    Affiche une barre de progression dans la console
    
    Args:
        current: Valeur actuelle
        total: Valeur totale
        prefix: Texte avant la barre
        suffix: Texte après la barre
        length: Longueur de la barre
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    
    if current == total:
        print()


def validate_similarity_score(score: float) -> str:
    """
    Valide et catégorise un score de similarité
    
    Args:
        score: Score de similarité (0-1)
        
    Returns:
        Catégorie de similarité
    """
    if score >= config.SIMILARITY_THRESHOLD_HIGH:
        return "Haute similarité"
    elif score >= config.SIMILARITY_THRESHOLD_MEDIUM:
        return "Similarité moyenne"
    elif score >= config.SIMILARITY_THRESHOLD_LOW:
        return "Faible similarité"
    else:
        return "Glissement sémantique"


def export_to_excel(dataframes: Dict[str, pd.DataFrame], filename: str):
    """
    Exporte plusieurs DataFrames dans un fichier Excel avec plusieurs feuilles
    
    Args:
        dataframes: Dict {nom_feuille: DataFrame}
        filename: Nom du fichier Excel
    """
    filepath = config.TABLES_DIR / filename
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    logging.info(f"Excel exporté: {filepath}")
    return filepath


class Timer:
    """Classe pour mesurer le temps d'exécution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logging.info(f"{self.name} - Début")
        return self
    
    def __exit__(self, *args):
        import time
        elapsed_time = time.time() - self.start_time
        logging.info(f"{self.name} - Terminé en {elapsed_time:.2f}s")


if __name__ == "__main__":
    # Test des fonctions utilitaires
    logger = setup_logging()
    logger.info("Test du module utils.py")
    
    # Créer les dossiers
    create_directories()
    
    # Tester la sauvegarde/chargement
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    save_results(test_data, "test_utils", format='json')
    loaded_data = load_results("test_utils", format='json')
    print("Données chargées:", loaded_data)
    
    # Tester les statistiques de texte
    sample_text = "Ceci est un test. Il contient plusieurs phrases."
    stats = text_statistics(sample_text)
    print("Statistiques du texte:", stats)
    
    # Tester la validation de similarité
    print("Score 0.9:", validate_similarity_score(0.9))
    print("Score 0.6:", validate_similarity_score(0.6))
    print("Score 0.3:", validate_similarity_score(0.3))