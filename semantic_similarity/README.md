# Application de comparaison de similarité sémantique

Cette application permet de comparer la similarité sémantique entre un texte d'entrée et 10 textes de référence en utilisant trois méthodes différentes : FAISS, un calcul simple (Jaccard), et OpenAI.

## Installation

1. Clonez ce dépôt
2. Créez un environnement virtuel : `python -m venv .venv`
3. Activez l'environnement virtuel : `source .venv/bin/activate` (Linux/macOS) ou `.venv\Scripts\activate` (Windows)
4. Installez les dépendances : `pip install -r requirements.txt`
5. Remplacez 'your-api-key-here' dans `app.py` par votre véritable clé API OpenAI.

## Utilisation

1. Lancez l'application : `python app.py`
2. Ouvrez votre navigateur et accédez à `http://localhost:5000`
3. Entrez un texte dans la zone de texte et cliquez sur "Comparer avec tous les systèmes"
4. Les résultats de similarité pour chaque méthode seront affichés sous forme de tableaux

## Structure du projet

- `app.py`: Contient la logique principale de l'application Flask et les méthodes de calcul de similarité
- `templates/index.html`: L'interface utilisateur de l'application
- `requirements.txt`: Liste des dépendances Python nécessaires
- `.gitignore`: Fichiers et dossiers à ignorer par Git

## Contribution

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request
