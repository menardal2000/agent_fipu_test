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

# Exemple d'utilisation
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Tu es un assistant qui aide à visualiser des données."},
        {"role": "user", "content": "Donne moi les chiffres du déflateur du PIB en France disponibles dans le document RESF 2021"}
    ]

    response = generate_with_ollama(messages=messages, tools=tools, tool_choice="any")
    if "message" not in response:
        raise ValueError("La réponse d'Ollama ne contient pas de message")

    message = response["message"]
    if "tool_calls" in message and len(message["tool_calls"]) > 0:
        tool_call = message["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        function_params = tool_call["function"]["arguments"]
    else:
        raise ValueError("Pas d'appel de fonction dans la réponse")

    data = names_to_functions[function_name](**function_params)
    messages.append({
        "role": "tool",
        "content": json.dumps(data),
        "tool_call_id": "default_id"
    })

    messages.append({"role": "assistant", "content": "J'ai récupéré les données. Que souhaitez-vous faire avec ces données ?"})
    messages.append({"role": "user", "content": "Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années"})

    best_tool = find_best_visualization_tool("Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années")
    viz_params = extract_visualization_params("Sors un histogramme des chiffres d'inflation en ordonnée et en abscisse les années")

    if best_tool == "create_line_plot":
        create_line_plot(data_series=data, **viz_params)
    else:
        create_histogram(data_series=data, bins=len(data), **viz_params) 