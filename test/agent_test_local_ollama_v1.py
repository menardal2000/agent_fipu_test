import os
import torch
from pydantic import BaseModel, Field
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import functools
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration d'Ollama
OLLAMA_API_BASE = "http://localhost:11434/api"
MODEL_NAME = "mistral"

# Chargement du modèle d'embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Génère l'embedding d'un texte"""
    return embedding_model.encode(text)

def find_best_matching_column(query, columns):
    """Trouve la colonne la plus similaire à la requête en utilisant les embeddings"""
    query_embedding = get_embedding(query)
    column_embeddings = [get_embedding(col) for col in columns]
    
    # Calcul des similarités
    similarities = cosine_similarity([query_embedding], column_embeddings)[0]
    
    # Trouver l'index de la plus grande similarité
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    
    print(f"Meilleure correspondance trouvée : {columns[best_match_idx]} avec un score de {best_match_score:.2f}")
    return columns[best_match_idx]

def generate_with_ollama(messages, tools=None, tool_choice=None):
    """
    Fonction pour générer une réponse avec Ollama
    """
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
    response_data = response.json()
    return response_data

path = "C:/Users/menar/Documents/projet_agent_fipu/output/tables"

# fonction pour extraire les données d'inflation
def get_inflation_data(document, year, figures):
    # Nettoyage du nom du document
    document = document.split()[0]  # Prend seulement la première partie (RESF)
    
    # Construction du chemin
    path = f"C:/Users/menar/Documents/projet_agent_fipu/output/tables/Tableau_{document}_{year}_1.xlsx"
    print(f"Tentative d'ouverture du fichier : {path}")
    df = pd.read_excel(path)
    print("Colonnes disponibles :", df.columns.tolist())
    
    # Recherche de la meilleure correspondance avec les embeddings
    best_matching_column = find_best_matching_column(figures, df.columns.tolist())
    
    # Convertir la colonne Année en entier
    df["Annee"] = df["Annee"].astype(int)
    
    # Sélectionner la colonne demandée
    result_df = df[["Annee", best_matching_column]]
    
    # Convertir en liste de dictionnaires pour la sérialisation JSON
    return result_df.to_dict('records')

def create_histogram(data_series, title="Histogramme", xlabel="Valeurs", ylabel="Fréquence", bins=10):
    """
    Crée un histogramme à partir d'une série de données.
    
    Args:
        data_series: Liste de dictionnaires contenant les données
        title: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
        bins: Nombre de classes pour l'histogramme
    """
    # Extraire les années et les valeurs
    years = [item['Annee'] for item in data_series]
    values = [item[list(item.keys())[1]] for item in data_series]  # Prend la deuxième clé (la valeur)
    
    plt.figure(figsize=(10, 6))
    plt.bar(years, values, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(years)  # Affiche toutes les années sur l'axe x
    plt.show()

def create_line_plot(data_series, title="Graphique", xlabel="Valeurs", ylabel="Fréquence"):
    """
    Crée un graphique de courbes à partir d'une série de données.
    
    Args:
        data_series: Liste de dictionnaires contenant les données
        title: Titre du graphique
        xlabel: Label de l'axe x
        ylabel: Label de l'axe y
    """
    # Extraire les années et les valeurs
    years = [item['Annee'] for item in data_series]
    values = [item[list(item.keys())[1]] for item in data_series]  # Prend la deuxième clé (la valeur)
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(years)  # Affiche toutes les années sur l'axe x
    plt.show()

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

def find_best_visualization_tool(user_request):
    """
    Détermine le meilleur outil de visualisation en fonction de la demande de l'utilisateur
    en utilisant la similarité des embeddings.
    """
    # Dictionnaire des outils de visualisation et leurs descriptions
    visualization_tools = {
        "create_histogram": "histogramme graphique en barres distribution",
        "create_line_plot": "graphique de courbes évolution tendance série temporelle"
    }
    
    # Créer les embeddings pour la requête de l'utilisateur
    user_embedding = get_embedding(user_request)
    
    # Créer les embeddings pour chaque outil
    tool_embeddings = {name: get_embedding(desc) for name, desc in visualization_tools.items()}
    
    # Calculer les similarités
    similarities = {
        name: cosine_similarity([user_embedding], [embedding])[0][0]
        for name, embedding in tool_embeddings.items()
    }
    
    # Trouver l'outil avec la plus grande similarité
    best_tool = max(similarities.items(), key=lambda x: x[1])
    print(f"Outil choisi : {best_tool[0]} avec un score de similarité de {best_tool[1]:.2f}")
    
    return best_tool[0]

def extract_visualization_params(user_request):
    """
    Extrait les paramètres de personnalisation du graphique à partir de la demande de l'utilisateur.
    """
    # Dictionnaire des mots-clés pour chaque paramètre
    param_keywords = {
        "title": ["titre", "nom", "intitulé", "évolution", "variation"],
        "xlabel": ["abscisse", "axe x", "axe horizontal", "années"],
        "ylabel": ["ordonnée", "axe y", "axe vertical", "valeur"]
    }
    
    # Créer les embeddings pour la requête
    request_embedding = get_embedding(user_request)
    
    # Initialiser les paramètres par défaut
    params = {
        "title": "Évolution des données",
        "xlabel": "Années",
        "ylabel": "Valeur"
    }
    
    # Pour chaque paramètre, chercher les mots-clés dans la requête
    for param, keywords in param_keywords.items():
        # Créer les embeddings pour les mots-clés
        keyword_embeddings = [get_embedding(kw) for kw in keywords]
        
        # Calculer les similarités
        similarities = [cosine_similarity([request_embedding], [kw_emb])[0][0] for kw_emb in keyword_embeddings]
        
        # Si une similarité significative est trouvée, extraire le texte après le mot-clé
        if max(similarities) > 0.5:
            # Trouver le mot-clé le plus similaire
            best_keyword_idx = np.argmax(similarities)
            best_keyword = keywords[best_keyword_idx]
            
            # Extraire le texte après le mot-clé
            if best_keyword in user_request.lower():
                # Trouver la position après le mot-clé
                pos = user_request.lower().find(best_keyword) + len(best_keyword)
                # Extraire le texte jusqu'au prochain point ou virgule
                next_stop = min(
                    user_request.find(".", pos) if user_request.find(".", pos) != -1 else len(user_request),
                    user_request.find(",", pos) if user_request.find(",", pos) != -1 else len(user_request)
                )
                extracted_text = user_request[pos:next_stop].strip()
                if extracted_text:
                    params[param] = extracted_text
    
    return params

messages = [
    {"role": "system", "content": """Tu es un assistant qui aide à visualiser des données. 
    Pour créer une visualisation, tu dois d'abord récupérer les données avec l'outil 'get_data', 
    puis utiliser l'outil approprié pour la visualisation."""},
    {"role": "user", "content": "Donne moi les chiffres du déflateur du PIB en France disponibles dans le document RESF 2021"}
]

response = generate_with_ollama(
    messages=messages,
    tools=tools,
    tool_choice="any"
)

# Vérification de la structure de la réponse
if "message" not in response:
    raise ValueError("La réponse d'Ollama ne contient pas de message")

message = response["message"]

# Extraction des informations de la réponse
if "tool_calls" in message and len(message["tool_calls"]) > 0:
    tool_call = message["tool_calls"][0]
    function_name = tool_call["function"]["name"]
    function_params = tool_call["function"]["arguments"]
else:
    raise ValueError("Pas d'appel de fonction dans la réponse")

data = names_to_functions[function_name](**function_params)

messages.append(
    {
        "role": "tool",
        "content": json.dumps(data),
        "tool_call_id": "default_id"
    }
)

# Nouvelle requête pour la visualisation
messages.append(
    {"role": "assistant", "content": "J'ai récupéré les données. Que souhaitez-vous faire avec ces données ?"}
)
messages.append(
    {"role": "user", "content": "Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années"}
)

# Déterminer le meilleur outil de visualisation
best_tool = find_best_visualization_tool("Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années")

# Extraire les paramètres de personnalisation
viz_params = extract_visualization_params("Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années")

# Créer la visualisation avec l'outil choisi
if best_tool == "create_line_plot":
    create_line_plot(
        data_series=data,
        title=viz_params["title"],
        xlabel=viz_params["xlabel"],
        ylabel=viz_params["ylabel"]
    )
else:
    create_histogram(
        data_series=data,
        title=viz_params["title"],
        xlabel=viz_params["xlabel"],
        ylabel=viz_params["ylabel"],
        bins=len(data)
    )









