from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import openai
import time
from queue import Queue
import threading
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

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

# Initialisation du modèle et du tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Fonction pour obtenir les embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Encodage des textes de référence
reference_vectors = get_embeddings(reference_texts)

# Fonction pour normaliser manuellement les vecteurs
def manual_normalize_L2(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Éviter la division par zéro
    norms[norms == 0] = 1
    return vectors / norms

# Encodage des textes de référence
reference_vectors = get_embeddings(reference_texts)

# Vérification et normalisation des vecteurs
if np.isnan(reference_vectors).any() or np.isinf(reference_vectors).any():
    print("Warning: NaN or Inf values detected in reference vectors. Replacing with zeros.")
    reference_vectors = np.nan_to_num(reference_vectors)

normalized_vectors = manual_normalize_L2(reference_vectors)

# Initialisation de l'index FAISS
dimension = normalized_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # Utiliser le produit scalaire au lieu de L2
index.add(normalized_vectors.astype('float32'))

# Vérification
print(f"Nombre de vecteurs dans l'index FAISS : {index.ntotal}")


# Queue pour stocker les mises à jour
update_queue = Queue()

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    return len(set1.intersection(set2)) / len(set1.union(set2))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/compare_all', methods=['GET'])
def compare_all():
    def update_progress(message):
        update_queue.put(message)

    update_progress("Début de la comparaison")
    start_time = time.time()
    
    update_progress("Début de la comparaison FAISS")
    # FAISS comparison
    faiss_results = []
    for i, query_vector in enumerate(reference_vectors):
        update_progress(f"Comparaison FAISS: vecteur {i+1}")
        query_vector = manual_normalize_L2(query_vector.reshape(1, -1).astype('float32'))
        if query_vector is None or np.isnan(query_vector).any() or np.isinf(query_vector).any():
            print(f"Warning: Invalid query vector for index {i}. Skipping.")
            faiss_results.append([{"id": str(j), "text": reference_texts[j], "similarity": 0.0} for j in range(len(reference_texts))])
            continue
        try:
            D, I = index.search(query_vector, len(reference_texts))
            faiss_results.append([
                {"id": str(j), "text": reference_texts[j], "similarity": float((d + 1) / 2)}  # Normaliser de -1..1 à 0..1
                for j, d in zip(I[0], D[0])
            ])
        except Exception as e:
            print(f"Error in FAISS search for vector {i}: {str(e)}")
            faiss_results.append([{"id": str(j), "text": reference_texts[j], "similarity": 0.0} for j in range(len(reference_texts))])
    faiss_time = time.time() - start_time
    update_progress(f"Comparaison FAISS terminée en {faiss_time:.2f} secondes")


    
    update_progress("Début de la comparaison Jaccard")
    # Simple Jaccard comparison
    jaccard_start_time = time.time()
    jaccard_results = []
    for i, text in enumerate(reference_texts):
        similarities = [
            {"id": str(j), "text": other_text, "similarity": jaccard_similarity(text, other_text)}
            for j, other_text in enumerate(reference_texts)
        ]
        jaccard_results.append(sorted(similarities, key=lambda x: x['similarity'], reverse=True))
    jaccard_time = time.time() - jaccard_start_time
    update_progress(f"Comparaison Jaccard terminée en {jaccard_time:.2f} secondes")
    
    update_progress("Début de la comparaison OpenAI")
    # OpenAI comparison
    openai_start_time = time.time()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai_results = []
    for i, text in enumerate(reference_texts):
        similarities = []
        for j, other_text in enumerate(reference_texts):
            if i != j:
                update_progress(f"Comparaison OpenAI: texte {i+1} avec texte {j+1}")
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that rates the semantic similarity between two texts on a scale of 0 to 1."},
                            {"role": "user", "content": f"Rate the semantic similarity between these two texts on a scale of 0 to 1, where 1 is identical meaning and 0 is completely unrelated. Only respond with a number between 0 and 1.\n\nText 1: {text}\n\nText 2: {other_text}"}
                        ],
                        max_tokens=1,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                    raw_response = response.choices[0].message['content'].strip()
                    print(f"OpenAI raw response for texts {i+1} and {j+1}: {raw_response}")
                    similarity = float(raw_response)
                    # Assurez-vous que la similarité est entre 0 et 1
                    similarity = max(0, min(1, similarity))
                except Exception as e:
                    print(f"Error in OpenAI comparison for texts {i+1} and {j+1}: {str(e)}")
                    similarity = 0  # Valeur par défaut en cas d'erreur
                similarities.append({"id": str(j), "text": other_text, "similarity": similarity})
            else:
                similarities.append({"id": str(j), "text": other_text, "similarity": 1.0})
        openai_results.append(sorted(similarities, key=lambda x: x['similarity'], reverse=True))
    openai_time = time.time() - openai_start_time
    update_progress(f"Comparaison OpenAI terminée en {openai_time:.2f} secondes")
    
    # Calculate KPIs
    kpis = {
        "faiss": {
            "time": faiss_time,
            "avg_similarity": np.mean([result[1]['similarity'] for result in faiss_results]),  # Changé de [0] à [1]
            "min_similarity": min([result[-1]['similarity'] for result in faiss_results]),
            "max_similarity": max([result[1]['similarity'] for result in faiss_results if result[1]['id'] != str(i)])  # Changé de [0] à [1]
        },
        "jaccard": {
            "time": jaccard_time,
            "avg_similarity": np.mean([result[0]['similarity'] for result in jaccard_results if result[0]['id'] != str(i)]),
            "min_similarity": min([result[-1]['similarity'] for result in jaccard_results if result[-1]['id'] != str(i)]),
            "max_similarity": max([result[0]['similarity'] for result in jaccard_results if result[0]['id'] != str(i)])
        },
        "openai": {
            "time": openai_time,
            "avg_similarity": np.mean([result[0]['similarity'] for result in openai_results if result[0]['id'] != str(i)]),
            "min_similarity": min([result[-1]['similarity'] for result in openai_results if result[-1]['id'] != str(i)]),
            "max_similarity": max([result[0]['similarity'] for result in openai_results if result[0]['id'] != str(i)])
        }
    }
    
    total_time = time.time() - start_time
    update_progress("Comparaison terminée")
    update_progress(f"Temps total: {total_time:.2f} secondes")
    
    return jsonify({
        "faiss_results": faiss_results,
        "jaccard_results": jaccard_results,
        "openai_results": openai_results,
        "kpis": kpis,
        "total_time": total_time
    })

@app.route('/api/get_updates', methods=['GET'])
def get_updates():
    updates = []
    while not update_queue.empty():
        updates.append(update_queue.get())
    return jsonify({"updates": updates})

if __name__ == '__main__':
    app.run(debug=True)