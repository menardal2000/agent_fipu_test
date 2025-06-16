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
from datetime import datetime

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
    # On retourne la correspondance même si le score est bas, la vérification finale se fera dans find_best_column_match
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

def get_inflation_data(document, figures):
    # Extraction de l'année du document (ex: "RESF 2021" -> "2021")
    year = document.split()[-1]
    # Extraction du nom du document sans l'année (ex: "RESF 2021" -> "RESF")
    doc_name = document.split()[0]
    
    path = f"{DATA_PATH}/Tableau_{doc_name}_{year}.xlsx"
    df = pd.read_excel(path)
    best_matching_column, score = find_best_column_match(figures, df.columns.tolist())
    if best_matching_column is None:
        return None
    
    # Nettoyage des données
    df = df.replace([np.inf, -np.inf], np.nan)  # Remplacer inf par NaN
    df = df.dropna(subset=["Annee", best_matching_column])  # Supprimer les lignes avec NaN
    
    # Conversion en types numériques
    try:
        df["Annee"] = pd.to_numeric(df["Annee"], errors='coerce').astype('Int64')  # Int64 permet les valeurs NA
        df[best_matching_column] = pd.to_numeric(df[best_matching_column], errors='coerce')
        
        # Supprimer les lignes où la conversion a échoué
        df = df.dropna(subset=["Annee", best_matching_column])
        
        if df.empty:
            return None
            
        result_df = df[["Annee", best_matching_column]]
        return result_df.to_dict('records')
    except Exception as e:
        print(f"Erreur lors de la conversion des données : {str(e)}")
        return None

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
    en utilisant le LLM.
    """
    # Initialisation des paramètres
    params = {
        "figures": None,
        "document": None,
        "visualization_type": None,
        "visualization_params": {},
        "start_year": None,
        "end_year": None
    }
    
    # Préparation du prompt pour le LLM
    messages = [
        {
            "role": "system",
            "content": """Tu es un assistant spécialisé dans l'analyse des requêtes concernant les données économiques.
            Ta tâche est d'extraire les informations suivantes d'une requête :
            1. La série temporelle demandée (par exemple : 'déficit public', 'consommation intermédiaire', etc.)
            2. Le document source (RESF et année)
            3. Le type de visualisation demandé (histogramme ou graphique en courbes)
            4. L'année de début (peut être une année spécifique ou une expression comme "depuis 5 ans")
            5. L'année de fin (peut être une année spécifique ou l'année actuelle)
            
            IMPORTANT : Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans aucun texte supplémentaire.
            Format de réponse attendu :
            {
                "figures": "nom de la série demandée",
                "document": "RESF XXXX",
                "visualization_type": "histogram" ou "line_plot",
                "start_year": "année de début ou null",
                "end_year": "année de fin ou null"
            }
            
            Si une information n'est pas présente dans la requête, utilise null comme valeur.
            Pour les expressions comme "depuis n ans", calcule l'année de début en soustrayant n-1 de l'année actuelle (pour inclure l'année actuelle).
            Pour le type de visualisation, utilise "histogram" pour histogramme et "line_plot" pour graphique en courbes.
            Pour les années, utilise des nombres entiers (par exemple : 2020, 2021, etc.)."""
        },
        {
            "role": "user",
            "content": user_request
        }
    ]
    
    # Appel au LLM
    response = generate_with_ollama(messages)
    
    try:
        # Extraction des informations de la réponse du LLM
        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"]
            
            # Nettoyage de la réponse pour obtenir un JSON valide
            content = content.strip()
            
            # Suppression des marqueurs de code si présents
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Affichage de la réponse brute pour le débogage
            print("Réponse brute du LLM:", content)
            
            try:
                # Parsing du JSON
                extracted_params = json.loads(content)
                
                # Vérification des types de données
                for key in ["figures", "document", "visualization_type", "start_year", "end_year"]:
                    if key in extracted_params and extracted_params[key] == "null":
                        extracted_params[key] = None
                
                # Mise à jour des paramètres
                params.update(extracted_params)
                
                # Extraction des paramètres de visualisation si nécessaire
                if params["visualization_type"]:
                    params["visualization_params"] = extract_visualization_params(user_request)
                
            except json.JSONDecodeError as e:
                print(f"Erreur de parsing JSON : {str(e)}")
                print(f"Contenu problématique : {content}")
                return params
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de la requête : {str(e)}")
    
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
        # Seuil de similarité à 0.2
        if best_match[1] > 0.2:
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
        from datetime import datetime
        
        # Configuration du style
        plt.style.use('default')
        
        # Création de la figure
        plt.figure(figsize=(10, 6))
        
        # Extraction des données
        years = [d['Annee'] for d in data]
        values = [d[list(d.keys())[1]] for d in data]
        series_name = list(data[0].keys())[1]  # Nom de la série affichée
        
        # Filtrage des données selon les années de début et de fin
        if params.get("start_year") is not None or params.get("end_year") is not None:
            current_year = datetime.now().year
            start_year = params.get("start_year", min(years))
            end_year = params.get("end_year", current_year)
            
            # Si start_year est une expression comme "depuis 5 ans"
            if isinstance(start_year, str) and "depuis" in start_year.lower():
                try:
                    years_diff = int(''.join(filter(str.isdigit, start_year)))
                    start_year = end_year - years_diff  # +1 pour inclure l'année actuelle      
                except ValueError:
                    start_year = min(years)
            
            # Filtrage des données
            filtered_data = [(y, v) for y, v in zip(years, values) if start_year <= y <= end_year]
            if filtered_data:
                years, values = zip(*filtered_data)
            else:
                return "Aucune donnée disponible pour la période spécifiée."
        
        if visualization_type == "histogram":
            plt.bar(years, values, color='skyblue', edgecolor='black')
            plt.title(f"{series_name}", fontsize=12, pad=15)
            plt.xlabel("Année", fontsize=10)
            plt.ylabel("Valeur", fontsize=10)
            
        elif visualization_type == "line_plot":
            plt.plot(years, values, marker='o', linestyle='-', linewidth=2, color='blue')
            plt.title(f"Évolution de {series_name}", fontsize=12, pad=15)
            plt.xlabel("Année", fontsize=10)
            plt.ylabel("Valeur", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
        # Personnalisation
        plt.xticks(years, rotation=45)  # Force l'affichage des années exactes
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
            figures=params["figures"]
        )
        if not data:
            return "Je n'ai pas pu récupérer les données demandées."
        
        # Vérification de la correspondance des colonnes
        best_match, score = find_best_column_match(params["figures"], data[0].keys())
        if not best_match or score < 0.2:
            return f"Je n'ai pas pu trouver une correspondance précise pour '{params['figures']}'. Veuillez reformuler votre demande."
        
        # Création de la visualisation si demandée
        if params["visualization_type"]:
            # Ajout des paramètres d'années aux paramètres de visualisation
            visualization_params = params["visualization_params"]
            visualization_params["start_year"] = params["start_year"]
            visualization_params["end_year"] = params["end_year"] if params["end_year"] is not None else datetime.now().year
            
            viz_path = create_visualization(data, params["visualization_type"], visualization_params)
            if viz_path:
                return f"J'ai créé une visualisation des données. Vous pouvez la trouver dans : {viz_path}"
            else:
                return "Je n'ai pas pu créer la visualisation demandée."
        
        return "Veuillez spécifier le type de visualisation souhaité (histogramme ou graphique en courbes)."
        
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