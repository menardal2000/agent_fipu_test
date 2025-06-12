import os
import torch
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import functools
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
OLLAMA_API_BASE = "http://localhost:11434/api"
MODEL_NAME = "mistral"
DATA_PATH = "C:/Users/menar/Documents/projet_agent_fipu/output/tables"

# Modèle d'embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedding_model.encode(text)

def find_best_matching_column(query, columns):
    query_embedding = get_embedding(query)
    column_embeddings = [get_embedding(col) for col in columns]
    similarities = cosine_similarity([query_embedding], column_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    print(f"Meilleure correspondance : {columns[best_match_idx]} (score: {best_match_score:.2f})")
    return columns[best_match_idx]

def generate_with_ollama(messages, tools=None, tool_choice=None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    response = requests.post(f"{OLLAMA_API_BASE}/chat", json=payload)
    return response.json()

def get_inflation_data(document, year, figures):
    document = document.split()[0]
    path = f"{DATA_PATH}/Tableau_{document}_{year}_1.xlsx"
    print(f"Ouverture du fichier : {path}")
    df = pd.read_excel(path)
    print("Colonnes disponibles :", df.columns.tolist())
    best_matching_column = find_best_matching_column(figures, df.columns.tolist())
    df["Annee"] = df["Annee"].astype(int)
    result_df = df[["Annee", best_matching_column]]
    return result_df.to_dict('records')

def create_histogram(data_series, title="Histogramme", xlabel="Valeurs", ylabel="Fréquence", bins=10):
    years = [item['Annee'] for item in data_series]
    values = [item[list(item.keys())[1]] for item in data_series]
    plt.figure(figsize=(10, 6))
    plt.bar(years, values, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(years)
    plt.show()

def create_line_plot(data_series, title="Graphique", xlabel="Valeurs", ylabel="Fréquence"):
    years = [item['Annee'] for item in data_series]
    values = [item[list(item.keys())[1]] for item in data_series]
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(years)
    plt.show()

def find_best_visualization_tool(user_request):
    visualization_tools = {
        "create_histogram": "histogramme graphique en barres distribution",
        "create_line_plot": "graphique de courbes évolution tendance série temporelle"
    }
    user_embedding = get_embedding(user_request)
    tool_embeddings = {name: get_embedding(desc) for name, desc in visualization_tools.items()}
    similarities = {
        name: cosine_similarity([user_embedding], [embedding])[0][0]
        for name, embedding in tool_embeddings.items()
    }
    best_tool = max(similarities.items(), key=lambda x: x[1])
    print(f"Outil choisi : {best_tool[0]} (score: {best_tool[1]:.2f})")
    return best_tool[0]

def extract_visualization_params(user_request):
    param_keywords = {
        "title": ["titre", "nom", "intitulé", "évolution", "variation"],
        "xlabel": ["abscisse", "axe x", "axe horizontal", "années"],
        "ylabel": ["ordonnée", "axe y", "axe vertical", "valeur"]
    }
    request_embedding = get_embedding(user_request)
    params = {
        "title": "Évolution des données",
        "xlabel": "Années",
        "ylabel": "Valeur"
    }
    for param, keywords in param_keywords.items():
        keyword_embeddings = [get_embedding(kw) for kw in keywords]
        similarities = [cosine_similarity([request_embedding], [kw_emb])[0][0] for kw_emb in keyword_embeddings]
        if max(similarities) > 0.5:
            best_keyword_idx = np.argmax(similarities)
            best_keyword = keywords[best_keyword_idx]
            if best_keyword in user_request.lower():
                pos = user_request.lower().find(best_keyword) + len(best_keyword)
                next_stop = min(
                    user_request.find(".", pos) if user_request.find(".", pos) != -1 else len(user_request),
                    user_request.find(",", pos) if user_request.find(",", pos) != -1 else len(user_request)
                )
                extracted_text = user_request[pos:next_stop].strip()
                if extracted_text:
                    params[param] = extracted_text
    return params

def analyze_user_request(user_request):
    """
    Analyse la requête de l'utilisateur pour extraire les informations nécessaires
    en utilisant uniquement la similarité sémantique.
    """
    # Initialisation des paramètres
    params = {
        "figures": None,
        "year": None,
        "document": None,
        "visualization_type": None,
        "visualization_params": {}
    }
    
    # Extraction des informations de base
    request_embedding = get_embedding(user_request)
    
    # Extraction de l'année avec regex (nécessaire pour la précision)
    import re
    year_match = re.search(r'\b(20\d{2})\b', user_request)
    if year_match:
        params["year"] = year_match.group(1)
    
    # Extraction du document RESF avec regex (nécessaire pour la précision)
    resf_match = re.search(r'RESF\s*\d{4}', user_request)
    if resf_match:
        params["document"] = resf_match.group()
    
    # Extraction des figures demandées
    # On divise la requête en segments pour trouver la partie qui décrit les figures
    segments = user_request.split()
    best_segment = None
    best_score = 0.0
    
    # Exemples de figures pour la comparaison sémantique
    figure_examples = [
        "croissance du PIB",
        "croissance réelle du PIB",
        "déflateur du PIB",
        "solde public",
        "déficit public"
    ]
    
    # Calculer la similarité pour chaque segment
    for i in range(len(segments)):
        for j in range(i + 1, len(segments) + 1):
            segment = " ".join(segments[i:j])
            segment_embedding = get_embedding(segment)
            
            # Calculer la similarité avec chaque exemple
            similarities = []
            for example in figure_examples:
                example_embedding = get_embedding(example)
                similarity = cosine_similarity([segment_embedding], [example_embedding])[0][0]
                similarities.append(similarity)
            
            # Garder le meilleur score
            max_similarity = max(similarities)
            if max_similarity > best_score:
                best_score = max_similarity
                best_segment = segment
    
    # Si on a trouvé un segment avec une bonne similarité
    if best_score > 0.3:  # Seuil bas pour capturer plus de variations
        params["figures"] = best_segment
    
    # Extraction du type de visualisation
    viz_examples = {
        "histogram": [
            "sous forme d'histogramme",
            "en barres",
            "graphique en barres",
            "distribution"
        ],
        "line_plot": [
            "sous forme de courbe",
            "en courbe",
            "graphique en courbe",
            "évolution"
        ]
    }
    
    viz_embeddings = {
        viz_type: [get_embedding(ex) for ex in examples]
        for viz_type, examples in viz_examples.items()
    }
    
    viz_similarities = {
        viz_type: max([cosine_similarity([request_embedding], [emb])[0][0] for emb in embeddings])
        for viz_type, embeddings in viz_embeddings.items()
    }
    
    if max(viz_similarities.values()) > 0.5:
        params["visualization_type"] = max(viz_similarities.items(), key=lambda x: x[1])[0]
        params["visualization_params"] = extract_visualization_params(user_request)
    
    # Si certaines informations sont manquantes, essayer de les déduire du contexte
    if not params["document"] and params["year"]:
        params["document"] = f"RESF {params['year']}"
    
    return params

def find_best_column_match(figures, available_columns):
    """
    Trouve la meilleure correspondance entre la figure demandée et les colonnes disponibles
    en utilisant uniquement la similarité sémantique.
    """
    if not figures or not available_columns:
        return None, 0.0
    
    # Extraction des embeddings
    figures_embedding = get_embedding(figures)
    
    # Calcul des similarités pour chaque colonne
    similarities = {}
    for col in available_columns:
        if col == "Annee":  # Ignorer la colonne Année
            continue
        
        # Calcul de la similarité avec la colonne
        col_embedding = get_embedding(col)
        similarity = cosine_similarity([figures_embedding], [col_embedding])[0][0]
        similarities[col] = similarity
        
        # Affichage des similarités pour le débogage
        print(f"Similarité entre '{figures}' et '{col}': {similarity:.3f}")
    
    # Trouver la meilleure correspondance
    if similarities:
        best_match = max(similarities.items(), key=lambda x: x[1])
        if best_match[1] > 0.5:  # Seuil de similarité
            return best_match[0], best_match[1]
    
    return None, 0.0

# Configuration des outils
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Get the data of a country for a specific year",
            "parameters": {
                "type": "object",
                "properties": {
                    "figures": {"type": "string", "description": "What data do you want to get?"},
                    "year": {"type": "string", "description": "The year to get the data from"},
                    "document": {"type": "string", "description": "The document to get the data from"},
                },
                "required": ["figures", "year", "document"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_histogram",
            "description": "Create a histogram of the data",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_series": {"type": "array", "description": "The data series to visualize"},
                    "title": {"type": "string", "description": "Title of the histogram"},
                    "xlabel": {"type": "string", "description": "Label for x-axis"},
                    "ylabel": {"type": "string", "description": "Label for y-axis"},
                    "bins": {"type": "integer", "description": "Number of bins for the histogram"}
                },
                "required": ["data_series"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_line_plot",
            "description": "Create a line plot of the data",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_series": {"type": "array", "description": "The data series to visualize"},
                    "title": {"type": "string", "description": "Title of the line plot"},
                    "xlabel": {"type": "string", "description": "Label for x-axis"},
                    "ylabel": {"type": "string", "description": "Label for y-axis"}
                },
                "required": ["data_series"]
            }
        }
    }
]

names_to_functions = {
    "get_data": functools.partial(get_inflation_data),
    "create_histogram": functools.partial(create_histogram),
    "create_line_plot": functools.partial(create_line_plot)
}

def create_visualization(data, visualization_type, params):
    """
    Crée une visualisation des données selon le type spécifié.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Configuration du style
        plt.style.use('default')
        
        # Création de la figure
        plt.figure(figsize=(10, 6))
        
        # Extraction des données
        years = [d['Annee'] for d in data]
        values = [d[list(d.keys())[1]] for d in data]
        
        if visualization_type == "histogram":
            plt.bar(years, values, color='skyblue', edgecolor='black')
            plt.title(f"Distribution des valeurs par année", fontsize=12, pad=15)
            plt.xlabel("Année", fontsize=10)
            plt.ylabel("Valeur", fontsize=10)
            
        elif visualization_type == "line_plot":
            plt.plot(years, values, marker='o', linestyle='-', linewidth=2, color='blue')
            plt.title(f"Évolution des valeurs par année", fontsize=12, pad=15)
            plt.xlabel("Année", fontsize=10)
            plt.ylabel("Valeur", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
        # Personnalisation
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Création du dossier de sortie s'il n'existe pas
        import os
        os.makedirs("output/visualizations", exist_ok=True)
        
        # Sauvegarde du graphique
        output_path = "output/visualizations/graphique.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Erreur lors de la création de la visualisation : {str(e)}")
        return None

def process_user_request(user_request):
    """
    Traite la requête de l'utilisateur et génère la réponse appropriée.
    """
    try:
        # Analyse de la requête
        params = analyze_user_request(user_request)
        
        if not params["figures"] or not params["document"]:
            return "Je n'ai pas pu extraire toutes les informations nécessaires. Veuillez reformuler votre demande."
        
        # Récupération des données
        data = get_inflation_data(
            document=params["document"],
            year=params["year"],
            figures=params["figures"]
        )
        if not data:
            return "Je n'ai pas pu récupérer les données demandées."
        
        # Vérification de la correspondance des colonnes
        best_match, score = find_best_column_match(params["figures"], data[0].keys())
        if not best_match or score < 0.5:
            return f"Je n'ai pas pu trouver une correspondance précise pour '{params['figures']}'. Veuillez reformuler votre demande."
        
        # Création de la visualisation si demandée
        if params["visualization_type"]:
            viz_path = create_visualization(data, params["visualization_type"], params["visualization_params"])
            if viz_path:
                return f"J'ai créé une visualisation des données. Vous pouvez la trouver dans : {viz_path}"
        
        # Formatage de la réponse
        response = "Données récupérées :\n"
        for item in data:
            response += f"Année {item['Annee']}: {item[best_match]}\n"
        
        return response
        
    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}"

# Exemple d'utilisation
if __name__ == "__main__":
    print("Bienvenue ! Posez votre question sur les données économiques.")
    print("Pour quitter, tapez 'exit' ou 'quit'.")
    
    while True:
        try:
            user_input = input("\nVotre demande : ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Au revoir !")
                break
            
            response = process_user_request(user_input)
            print(response)
            
        except Exception as e:
            print(f"Une erreur s'est produite : {str(e)}")
            continue 