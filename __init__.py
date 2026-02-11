import sys
from pathlib import Path

# Ce code permet de s'assurer que la racine du projet est toujours 
# dans le chemin de recherche de Python, peu importe d'où on lance le script.
root = Path(__file__).parent.absolute()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))