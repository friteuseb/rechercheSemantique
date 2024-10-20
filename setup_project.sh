#!/bin/bash

# Création de la structure du projet
mkdir -p semantic_similarity/{templates,static}
cd semantic_similarity

# Création des fichiers Python principaux
touch app.py

# Création des fichiers de configuration et documentation
touch {README.md,requirements.txt,.gitignore}

# Contenu initial des fichiers

# app.py
cat << EOF > app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

app = Flask(__name__)
CORS(app)

# Les 10 textes de référence
reference_texts = [
    "Le chat noir dort sur le canapé.",
    "Un félin de couleur sombre se repose sur le sofa.",
    "Le chien brun aboie dans le jardin.",
    "La voiture rouge roule sur l'autoroute.",
    "Un véhicule écarlate se déplace sur une voie rapide.",
    "Le soleil brille dans le ciel bleu.",
    "L'astre du jour illumine le firmament azuré.",
    "La pomme verte est sur la table.",
    "Une pomme rouge est posée sur le bureau.",
    "Le programmeur écrit du code sur son ordinateur."
]

# Initialisation de FAISS
dimension = 384  # Doit correspondre à la dimension des vecteurs de votre modèle
index = faiss.IndexFlatL2(dimension)
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Encodage des textes de référence et ajout à l'index FAISS
text_vectors = model.encode(reference_texts)
index.add(np.array(text_vectors).astype('float32'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/find_similar', methods=['POST'])
def find_similar():
    data = request.json
    query_text = data['text']
    
    query_vector = model.encode([query_text])
    D, I = index.search(np.array(query_vector).astype('float32'), len(reference_texts))
    
    results = [
        {"id": str(i), "text": reference_texts[i], "similarity": float(1 - d)}
        for i, d in zip(I[0], D[0])
    ]
    
    return jsonify(results)

@app.route('/api/simple_similarity', methods=['POST'])
def simple_similarity():
    data = request.json
    query_text = data['text']
    
    def jaccard_similarity(text1, text2):
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    results = [
        {"id": str(i), "text": text, "similarity": jaccard_similarity(query_text, text)}
        for i, text in enumerate(reference_texts)
    ]
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify(results)

@app.route('/api/openai_similarity', methods=['POST'])
def openai_similarity():
    data = request.json
    query_text = data['text']
    
    openai.api_key = 'your-api-key-here'
    
    results = []
    for i, text in enumerate(reference_texts):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Rate the semantic similarity between these two texts on a scale of 0 to 1, where 1 is identical meaning and 0 is completely unrelated:\\n\\nText 1: {query_text}\\n\\nText 2: {text}\\n\\nSimilarity score:",
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0.5,
        )
        similarity = float(response.choices[0].text.strip())
        results.append({"id": str(i), "text": text, "similarity": similarity})
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
EOF

# index.html
cat << 'EOF' > templates/index.html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparaison de Similarité Sémantique</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Comparaison de Similarité Sémantique</h1>
        
        <div class="mb-3">
            <label for="textInput" class="form-label">Texte d'entrée :</label>
            <textarea class="form-control" id="textInput" rows="3"></textarea>
        </div>
        
        <div class="mb-3">
            <button id="compareAll" class="btn btn-success">Comparer avec tous les systèmes</button>
        </div>
        
        <div id="results" class="mt-4">
            <h2>Résultats :</h2>
            <div id="faissResults"></div>
            <div id="simpleResults"></div>
            <div id="openaiResults"></div>
        </div>
    </div>

    <script>
    function createTable(title, data) {
        let html = `<h3>${title}</h3>`;
        html += '<table class="table table-striped">';
        html += '<thead><tr><th>ID</th><th>Texte</th><th>Similarité</th></tr></thead>';
        html += '<tbody>';
        data.forEach(item => {
            html += `<tr><td>${item.id}</td><td>${item.text}</td><td>${item.similarity.toFixed(4)}</td></tr>`;
        });
        html += '</tbody></table>';
        return html;
    }

    $(document).ready(function() {
        $('#compareAll').click(function() {
            const text = $('#textInput').val();
            if (text) {
                // FAISS comparison
                $.ajax({
                    url: '/api/find_similar',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ text: text }),
                    success: function(response) {
                        $('#faissResults').html(createTable('Résultats FAISS', response));
                    },
                    error: function() {
                        $('#faissResults').html('<h3>Résultats FAISS :</h3><p>Erreur lors de la comparaison</p>');
                    }
                });

                // Simple comparison
                $.ajax({
                    url: '/api/simple_similarity',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ text: text }),
                    success: function(response) {
                        $('#simpleResults').html(createTable('Résultats Calcul Simple', response));
                    },
                    error: function() {
                        $('#simpleResults').html('<h3>Résultats Calcul Simple :</h3><p>Erreur lors de la comparaison</p>');
                    }
                });

                // OpenAI comparison
                $.ajax({
                    url: '/api/openai_similarity',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ text: text }),
                    success: function(response) {
                        $('#openaiResults').html(createTable('Résultats OpenAI', response));
                    },
                    error: function() {
                        $('#openaiResults').html('<h3>Résultats OpenAI :</h3><p>Erreur lors de la comparaison</p>');
                    }
                });
            }
        });
    });
    </script>
</body>
</html>
EOF

# requirements.txt
cat << EOF > requirements.txt
flask==2.0.1
flask-cors==3.0.10
faiss-cpu==1.7.1
numpy==1.21.0
sentence-transformers==2.1.0
openai==0.27.0
EOF

# .gitignore
cat << EOF > .gitignore
__pycache__/
*.pyc
.venv/
EOF

# README.md
cat << EOF > README.md
# Application de comparaison de similarité sémantique

Cette application permet de comparer la similarité sémantique entre un texte d'entrée et 10 textes de référence en utilisant trois méthodes différentes : FAISS, un calcul simple (Jaccard), et OpenAI.

## Installation

1. Clonez ce dépôt
2. Créez un environnement virtuel : \`python -m venv .venv\`
3. Activez l'environnement virtuel : \`source .venv/bin/activate\` (Linux/macOS) ou \`.venv\\Scripts\\activate\` (Windows)
4. Installez les dépendances : \`pip install -r requirements.txt\`
5. Remplacez 'your-api-key-here' dans \`app.py\` par votre véritable clé API OpenAI.

## Utilisation

1. Lancez l'application : \`python app.py\`
2. Ouvrez votre navigateur et accédez à \`http://localhost:5000\`
3. Entrez un texte dans la zone de texte et cliquez sur "Comparer avec tous les systèmes"
4. Les résultats de similarité pour chaque méthode seront affichés sous forme de tableaux

## Structure du projet

- \`app.py\`: Contient la logique principale de l'application Flask et les méthodes de calcul de similarité
- \`templates/index.html\`: L'interface utilisateur de l'application
- \`requirements.txt\`: Liste des dépendances Python nécessaires
- \`.gitignore\`: Fichiers et dossiers à ignorer par Git

## Contribution

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request
EOF

echo "Structure du projet et fichiers de base créés avec succès!"