# Projet-NLP
Intelligence artificielle et finance publique
# Projet NLP ISE3 - Analyse Sémantique des Lois de Finances du Cameroun

**Intelligence Artificielle et Finances Publiques : Utilisation des modèles Transformers pour l'audit sémantique et la classification budgétaire**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Table des Matières

1. [Description du Projet](#-description-du-projet)
2. [Objectifs](#-objectifs)
3. [Structure du Projet](#-structure-du-projet)
4. [Prérequis](#-prérequis)
5. [Installation](#-installation)
6. [Données Requises](#-données-requises)
7. [Exécution du Projet](#-exécution-du-projet)
8. [Résultats Attendus](#-résultats-attendus)
9. [Technologies Utilisées](#-technologies-utilisées)
10. [Auteurs](#-auteurs)
11. [Contact](#-contact)

---

##  Description du Projet

Ce projet vise à analyser les **Lois de Finances du Cameroun** (2024-2026) en utilisant des techniques avancées de **Traitement du Langage Naturel (NLP)** et des modèles **Transformers**. L'objectif est de mesurer l'évolution des priorités budgétaires de l'État et de vérifier leur alignement avec la **Stratégie Nationale de Développement 2020-2030 (SND30)**.

### Contexte

Dans le cadre de la SND30, la Loi de Finances constitue l'instrument pivot de la politique économique du Cameroun. La complexité et le volume des documents textuels rendent difficile l'évaluation rapide de la cohérence sémantique entre les budgets annuels et les objectifs stratégiques de l'État.

### Problématique

**Comment l'intelligence artificielle, à travers le NLP, peut-elle mesurer mathématiquement l'évolution des priorités de l'État camerounais entre la Loi de Finances 2024 et les perspectives de 2025-2026 ?**

Existe-t-il un alignement statistiquement significatif entre le discours budgétaire et les piliers de la SND30 ?

---

##  Objectifs

Le projet s'articule autour de **trois axes principaux** :

### 1.  Audit Sémantique par Embeddings
- Utiliser **Sentence-BERT** pour calculer la similarité cosinus entre les articles de loi 2024 et 2025
- Identifier les ruptures de discours et le "glissement sémantique"
- Visualiser l'évolution thématique entre les années

### 2. Classification Zero-Shot
- Classer automatiquement les lignes de dépenses dans les 4 piliers de la SND30 :
  - **Transformation structurelle**
  - **Capital humain**
  - **Gouvernance**
  - **Développement régional**

### 3.  Analyse Statistique de Conformité
- Corréler les fréquences thématiques extraites avec les montants financiers réels
- Effectuer des tests de significativité statistique
- Analyser l'alignement Budget-SND30

---

##  Structure du Projet

```
Projet-NLP/
│
├── README.md                          # Ce fichier
├── LICENSE                            # Licence du projet
├── .gitignore                         # Fichiers à ignorer par Git
├── requirements.txt                   # Dépendances Python
├── requirements_minimal.txt           # Dépendances minimales
├── config.py                          # Configuration centrale du projet
├── env_config.py                      # Gestion des variables d'environnement
├── .env.example                       # Template pour variables sensibles
├── test_config.py                     # Script de test de configuration
│
├── data/                              # Données du projet
│   ├── raw/                           # PDFs bruts (Lois de Finances)
│   │   ├── Loi_de_Finances_2024.pdf
│   │   ├── Loi_de_Finances_2025.pdf
│   │   |
│   │   └── SND30_Document.pdf
│   ├── processed/                     # Données nettoyées et prétraitées
│   │   ├── texts_2024.json
│   │   ├── texts_2025.json
│   │   └── texts_2026.json
│   └── results/                       # Résultats des analyses
│       ├── similarities.csv
│       ├── classifications.csv
│       └── statistics.json
│
├── src/                               # Code source principal
│   ├── __init__.py                    # Initialisation du package
│   ├── preprocessing.py               # Extraction et nettoyage des PDFs
│   ├── embeddings.py                  # Calculs de similarité sémantique
│   ├── classification.py              # Classification zero-shot
│   ├── statistical_analysis.py        # Analyses statistiques
│   └── utils.py                       # Fonctions utilitaires
│
├── models/                            # Modèles sauvegardés
│   ├── pretrained/                    # Modèles pré-entraînés téléchargés
│   └── finetuned/                     # Modèles fine-tunés (si applicable)
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_exploration.ipynb           # Exploration des données
│   ├── 02_preprocessing.ipynb         # Prétraitement détaillé
│   └── 03_modelisation.ipynb          # Modélisation et analyse
│
├── dashboard/                         # Application Streamlit
│   └── app.py                         # Dashboard interactif
│
├── outputs/                           # Outputs générés
│   ├── figures/                       # Graphiques et visualisations
│   ├── tables/                        # Tableaux de résultats
│   └── reports/                       # Rapports générés
│
├── tests/                             # Tests unitaires
│   └── test_preprocessing.py
│
└── logs/                              # Fichiers de logs
    └── nlp_project.log
```

---

##  Prérequis

### Logiciels
- **Python** : Version 3.8 ou supérieure
- **Git** : Pour cloner le repository
- **GPU (optionnel)** : NVIDIA GPU avec CUDA pour accélération (recommandé mais pas obligatoire)

### Connaissances
- Bases en Python
- Notions de NLP et Machine Learning
- Compréhension des modèles Transformers (recommandé)

---

##  Installation

### 1. Cloner le Repository

```bash
# Clonez le projet
git clone https://github.com/votre-username/Projet-NLP.git

# Naviguez dans le dossier
cd Projet-NLP
```

### 2. Créer l'Environnement Virtuel

```bash
# Créez un environnement virtuel nommé nlpenv
python -m venv nlpenv

# Activez l'environnement virtuel

# Sur Windows PowerShell :
.\nlpenv\Scripts\Activate.ps1

# Sur Windows CMD :
nlpenv\Scripts\activate.bat

# Sur Linux/Mac :
source nlpenv/bin/activate
```

**Note pour Windows** : Si vous obtenez une erreur de politique d'exécution PowerShell, exécutez :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Installer les Dépendances

**Option A : Installation complète (recommandée)**
```bash
# Mettez à jour pip
python -m pip install --upgrade pip

# Installez toutes les dépendances
pip install -r requirements.txt
```

**Option B : Installation minimale (si problèmes de chemins longs sur Windows)**
```bash
# Installez les dépendances essentielles
pip install -r requirements_minimal.txt
```

**Note** : Si vous rencontrez l'erreur de chemins trop longs sur Windows, consultez [ce guide](https://pip.pypa.io/warnings/enable-long-paths) pour activer les chemins longs.

### 4. Télécharger le Modèle SpaCy Français

```bash
python -m spacy download fr_core_news_md
```

### 5. Configurer les Variables d'Environnement (Optionnel)

```bash
# Copiez le fichier template
cp .env.example .env

# Éditez .env et ajoutez vos clés API si nécessaire
# HUGGINGFACE_TOKEN=votre_token_ici
```

### 6. Vérifier l'Installation

```bash
# Testez la configuration
python test_config.py
```

Vous devriez voir :
```
================================================================================
TESTING CONFIGURATION
================================================================================
[1/6] Testing config import...
✓ Config imported successfully
...
✓ All tests passed!
✓ Configuration is ready to use
```

---

##  Données Requises

### Documents à Placer dans `data/raw/`

Avant d'exécuter le projet, placez les documents suivants dans le dossier `data/raw/` :

1. **Loi_de_Finances_2024.pdf** - Loi de Finances 2024 du Cameroun
2. **Loi_de_Finances_2025.pdf** - Loi de Finances 2025 du Cameroun
3. **SND30_Document.pdf** - Document de la Stratégie Nationale de Développement 2020-2030

### Sources

Ces documents peuvent être téléchargés depuis :
- **Site du MINFI** : [https://www.minfi.cm](https://www.minfi.cm)
- **Site du MINEPAT** : [https://www.minepat.gov.cm](https://www.minepat.gov.cm)

---

##  Exécution du Projet

### Workflow Complet

Le projet doit être exécuté dans l'ordre suivant :

#### **Étape 1 : Exploration des Données** 📊

```bash
# Ouvrez le premier notebook
jupyter notebook notebooks/01_exploration.ipynb

# Ou lancez Jupyter Lab
jupyter lab
```

**Ce notebook permet de :**
- Charger et visualiser les PDFs
- Obtenir des statistiques descriptives
- Comprendre la structure des documents

#### **Étape 2 : Prétraitement** 🔧

```bash
# Exécutez le notebook de prétraitement
jupyter notebook notebooks/02_preprocessing.ipynb
```

**Ce notebook effectue :**
- Extraction de texte depuis les PDFs
- Nettoyage et normalisation
- Segmentation en phrases/paragraphes
- Sauvegarde dans `data/processed/`

**Ou exécutez directement le script :**
```bash
python -c "from src.preprocessing import preprocess_all_documents; preprocess_all_documents()"
```

#### **Étape 3 : Modélisation et Analyse** 🤖

```bash
# Exécutez le notebook de modélisation
jupyter notebook notebooks/03_modelisation.ipynb
```

**Ce notebook réalise :**
- Calcul des embeddings avec Sentence-BERT
- Analyse de similarité sémantique
- Classification zero-shot dans les piliers SND30
- Tests statistiques de conformité
- Génération des visualisations

#### **Étape 4 : Dashboard Interactif** 📈

```bash
# Lancez le dashboard Streamlit
streamlit run dashboard/app.py
```

Le dashboard s'ouvrira dans votre navigateur à l'adresse : `http://localhost:8501`

**Fonctionnalités du dashboard :**
- Baromètre de glissement sémantique 2024-2026
- Visualisation des similarités entre années
- Distribution des budgets par pilier SND30
- Statistiques de conformité
- Graphiques interactifs

---

##  Résultats Attendus

### 1. Rapport Technique

Un document détaillant :
- Architecture du modèle utilisé
- Métriques de performance (F1-score, Precision, Recall, Log-Loss)
- Analyses des lois de finances
- Tests statistiques effectués
- Interprétation des résultats

**Localisation** : `outputs/reports/rapport_technique.pdf`

### 2. Code Source Documenté

- Disponible sur GitHub
- Code commenté et structuré
- Reproductible

### 3. Dashboard Interactif

- Baromètre de glissement sémantique
- Visualisations dynamiques
- Métriques en temps réel

### 4. Fichiers de Résultats

Dans `data/results/` :
- `similarities.csv` : Matrice de similarité entre documents
- `classifications.csv` : Classification des dépenses par pilier
- `statistics.json` : Résultats des tests statistiques
- `semantic_drift.json` : Mesures de glissement sémantique

### 5. Visualisations

Dans `outputs/figures/` :
- Heatmaps de similarité
- Graphiques de distribution budgétaire
- Clustering des thèmes
- Évolution temporelle des priorités

---

##  Technologies Utilisées

### NLP et Deep Learning
- **Transformers** (Hugging Face) - Modèles CamemBERT, Sentence-BERT
- **PyTorch** - Framework de deep learning
- **SpaCy** - Traitement du langage naturel pour le français

### Data Science
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Scikit-learn** - Machine learning et statistiques
- **SciPy** - Analyses statistiques avancées

### Clustering
- **HDBSCAN** - Clustering hiérarchique
- **UMAP** - Réduction de dimensionnalité

### Visualisation
- **Matplotlib** - Graphiques statiques
- **Seaborn** - Visualisations statistiques
- **Plotly** - Graphiques interactifs

### Dashboard
- **Streamlit** - Interface web interactive

### Traitement PDF
- **pdfplumber** - Extraction de texte depuis PDFs
- **PyPDF2** - Manipulation de fichiers PDF

---

##  Métriques de Performance

Le projet évalue les modèles selon :

- **F1-Score** : Mesure de précision et rappel
- **Precision** : Exactitude des classifications
- **Recall** : Couverture des classifications
- **Accuracy** : Taux de réussite global
- **Log-Loss** : Perte logarithmique
- **Cosine Similarity** : Similarité sémantique (0-1)

**Seuils définis** :
- Similarité haute : > 0.85
- Similarité moyenne : 0.70 - 0.85
- Similarité faible : < 0.70
- Glissement sémantique : < 0.50

---

##  Tests

### Exécuter les Tests

```bash
# Tous les tests
pytest tests/

# Test spécifique
pytest tests/test_preprocessing.py

# Avec couverture
pytest --cov=src tests/
```

### Tests Disponibles

- `test_preprocessing.py` - Tests d'extraction et nettoyage PDF
- `test_embeddings.py` - Tests de calcul de similarité
- `test_classification.py` - Tests de classification
- `test_config.py` - Tests de configuration

---

## 📝 Critères d'Évaluation

Le projet sera noté selon :

### 1. Démarche Scientifique (40%)
- Originalité de l'approche
- Rigueur méthodologique
- Justification des choix techniques

### 2. Reproductibilité du Code (30%)
- Clarté du code
- Documentation
- Possibilité de reproduire les résultats

### 3. Qualité d'Écriture (30%)
- Clarté du rapport
- Pertinence des analyses
- Synthèse et conclusions

---

##  Contribution

### Guide de Contribution

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Commitez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Pushez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

### Standards de Code

- Suivez **PEP 8** pour Python
- Documentez toutes les fonctions
- Ajoutez des tests pour les nouvelles fonctionnalités
- Utilisez des noms de variables explicites

---

## 👥 Auteurs

**Groupe ISE3 - Promotion 2025-2026**

- **Nom Prénom 1** - [Email](mailto:email1@example.com)
- **Nom Prénom 2** - [Email](mailto:email2@example.com)
- **Nom Prénom 3** - [Email](mailto:email3@example.com)
- **Nom Prénom 4** - [Email](mailto:email4@example.com)

**Encadrant**
- **MBIA NDI Marie Thérèse** - [mbialaura12@gmail.com](mailto:mbialaura12@gmail.com)

---

##  Contact

Pour toute question concernant le projet :

- **Email** : mbialaura12@gmail.com
- **Institution** : ISSEA - Yaoundé
- **Date de soumission** : 17 février 2026

---

##  Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

##  Remerciements

- **ISSEA Yaoundé** pour le cadre pédagogique
- **Hugging Face** pour les modèles Transformers
- **MINFI & MINEPAT** pour les données publiques
- La communauté **Python NLP** pour les outils open-source

---

##  Bibliographie

1. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805

2. Martin, L. et al. (2020). *CamemBERT: a Tasty French Language Model*. Proceedings of ACL 2020.

3. République du Cameroun (2020). *Document de Stratégie Nationale de Développement 2020-2030 (SND30)*.

4. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP 2019.

---

##  Liens Utiles

- [Documentation Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Documentation SpaCy](https://spacy.io/usage)
- [Guide PyTorch](https://pytorch.org/tutorials/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Site MINFI Cameroun](https://www.minfi.cm)

---

##  Configuration Système Recommandée

### Minimum
- CPU : Intel Core i5 ou équivalent
- RAM : 8 GB
- Stockage : 10 GB disponible
- OS : Windows 10/11, Ubuntu 20.04+, macOS 10.15+

### Recommandé
- CPU : Intel Core i7 ou équivalent
- RAM : 16 GB
- GPU : NVIDIA GPU avec 6GB+ VRAM
- Stockage : 20 GB disponible (SSD recommandé)

---

##  Problèmes Connus et Solutions

### Problème : Chemins trop longs sur Windows
**Solution** : Activez les chemins longs Windows ([Guide](https://pip.pypa.io/warnings/enable-long-paths))

### Problème : PyTorch n'utilise pas le GPU
**Solution** : 
```bash
# Vérifiez la disponibilité CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Installez la version CUDA de PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problème : SpaCy model non trouvé
**Solution** :
```bash
python -m spacy download fr_core_news_md
```

### Problème : Erreur d'import de modules
**Solution** :
```bash
# Vérifiez que l'environnement virtuel est activé
# Réinstallez les dépendances
pip install -r requirements.txt --force-reinstall
```

---

##  État du Projet

- [x] Structure du projet
- [x] Configuration de base
- [ ] Prétraitement des données
- [ ] Implémentation des embeddings
- [ ] Classification zero-shot
- [ ] Analyses statistiques
- [ ] Dashboard Streamlit
- [ ] Rapport technique
- [ ] Tests unitaires

---

**Dernière mise à jour** : Février 2026

**Version** : 1.0.0

---

*Développé par les étudiants ISE3 de l'ISSEA*