import os
import torch
from credentials.key import key_Mistral
from pydantic import BaseModel, Field
from mistralai import Mistral
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import functools
import json


# lancement de la connexion à l'api de Mistral
api_key = key_Mistral
client = Mistral(api_key=api_key)

path = "C:/Users/menar/Documents/projet_agent_fipu/output/tables"

# fonction pour extraire les données d'inflation
def get_inflation_data(document, year, figures):
    path = f"C:/Users/menar/Documents/projet_agent_fipu/output/tables/Tableau_{document}_{year}_1.xlsx" 
    df = pd.read_excel(path)
    print("Colonnes disponibles :", df.columns.tolist())
    
    # Recherche insensible à la casse
    matching_columns = [col for col in df.columns if col.lower() == figures.lower()]
    if not matching_columns:
        raise ValueError(f"La colonne '{figures}' n'a pas été trouvée. Colonnes disponibles : {df.columns.tolist()}")
    
    # Convertir la colonne Année en entier
    df["Annee"] = df["Annee"].astype(int)
    
    # Sélectionner la colonne demandée
    result_df = df[["Annee", matching_columns[0]]]
    
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
    }
]


names_to_functions = {
    "get_data": functools.partial(get_inflation_data),
    "create_histogram": functools.partial(create_histogram)
}

messages = [
    {"role": "system", "content": "Extrais les informations à chercher et les outils à utiliser pour satisfaire la demande de l'utilisateur"},
    {"role": "user", "content": "Donne moi les chiffres du déflateur du PIB en France disponibles dans le document RESF 2021"}
]

response = client.chat.complete(
    model = "mistral-large-latest",
    messages = messages,
    tools=tools,
    tool_choice="any",
    parallel_tool_calls = False
)

messages.append(response.choices[0].message)

tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

data = names_to_functions[function_name](**function_params)

messages.append(
    {
        "role": "tool",
        "content": json.dumps(data),
        "tool_call_id": tool_call.id
    }
)

# Nouvelle requête pour l'histogramme
messages.append(
    {"role": "assistant", "content": "J'ai récupéré les données. Que souhaitez-vous faire avec ces données ?"}
)
messages.append(
    {"role": "user", "content": "Sors un histogramme des chiffres d'inflation par année"}
)

response = client.chat.complete(
    model = "mistral-large-latest",
    messages = messages,
    tools=tools,
    tool_choice="any",
    parallel_tool_calls = False
)


# Traitement de la réponse pour créer l'histogramme
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

# Création de l'histogramme
create_histogram(
    data_series=data,  # Les données récupérées précédemment
    title="Évolution du déflateur du PIB en France",
    xlabel="Années",
    ylabel="Valeur du déflateur",
    bins=len(data)  # Un bin par année
)









