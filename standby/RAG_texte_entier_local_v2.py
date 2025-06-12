import os
import time
import torch
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from credentials.key import key_Mistral, HF_TOKEN
from parametres.params import parent_dir, input_dir, output_dir, database_dir, RESF_dir, docs_dir, tables_dir, txt_dir, graph_dir, report_dir
from mistralai import Mistral

print("🚀 Démarrage du programme...")

# ============================ Initialisation des clés API ============================
os.environ["HF_TOKEN"] = HF_TOKEN
login(os.environ["HF_TOKEN"])

# ============================ Initialisation des clients ============================
print("🔑 Initialisation du client Mistral...")
client = Mistral(api_key=key_Mistral)

print("🤖 Initialisation du modèle de chat Mistral...")
try:
    model_chat = ChatMistralAI(model="mistral-large-latest", api_key=key_Mistral)
    print("✅ Modèle de chat Mistral initialisé avec succès")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du modèle Mistral: {e}")
    exit()

# ============================ Modèle d'embeddings ============================
print("🤖 Initialisation du modèle d'embeddings...")
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

# ============================ Prompt Template ============================
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question} in French.
Provide a structured table of figures.
Don't justify your answers.
Do not say \"according to the context\" or \"mentioned in the context\" or similar.
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ============================ Chemins des dossiers ============================
path_input = os.path.join(parent_dir, input_dir)
path_resf = os.path.join(path_input, RESF_dir)
path_output = os.path.join(parent_dir, output_dir)
path_graph = os.path.join(path_output, graph_dir)
path_database = os.path.join(path_output, database_dir)
path_txt = os.path.join(path_output, txt_dir)
path_report = os.path.join(path_output, report_dir)
path_tables = os.path.join(path_output, tables_dir)

os.makedirs(path_input, exist_ok=True)
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_graph, exist_ok=True)
os.makedirs(path_database, exist_ok=True)
os.makedirs(path_txt, exist_ok=True)
os.makedirs(path_report, exist_ok=True)
os.makedirs(path_tables, exist_ok=True)

# ============================ Fonction de traitement OCR ============================
def process_ocr(annee):
    input_file = f"RESF_{annee}.pdf"
    input_path = os.path.join(path_resf, input_file)
    
    if not os.path.exists(input_path):
        print(f"⚠️ Fichier {input_file} non trouvé dans {input_path}")
        return None

    print(f"📄 Traitement du fichier {input_file}...")
    
    try:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": input_path,
                "content": open(input_path, "rb"),
            },
            purpose="ocr"
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )

        ocr_text_with_pages = ""
        for page_number, page in enumerate(ocr_response.pages, start=1):
            if page.markdown:
                ocr_text_with_pages += f"[{page_number}]\n{page.markdown}\n"

        file_name = f"RESF_{annee}.txt"
        file_path = os.path.join(path_txt, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(ocr_text_with_pages)
        
        print(f"✅ OCR terminé pour {input_file}")
        return file_path
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement OCR de {input_file}: {str(e)}")
        return None

# ============================ Fonctions de visualisation ============================
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
    
    # Sauvegarde du graphique
    graph_filename = f"histogramme_{title.replace(' ', '_')}.png"
    graph_path = os.path.join(path_graph, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Graphique sauvegardé : {graph_path}")
    return graph_path

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
    
    # Sauvegarde du graphique
    graph_filename = f"courbe_{title.replace(' ', '_')}.png"
    graph_path = os.path.join(path_graph, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Graphique sauvegardé : {graph_path}")
    return graph_path

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

# ============================ Analyse de la demande ============================
def analyze_user_request(user_request):
    """
    Analyse la requête de l'utilisateur pour extraire les informations nécessaires
    """
    params = {
        "figures": None,
        "year": None,
        "document": None,
        "visualization_type": None,
        "visualization_params": {}
    }
    
    # Extraction de l'année
    year_match = re.search(r'\b(20\d{2})\b', user_request)
    if year_match:
        params["year"] = year_match.group(1)
    
    # Extraction du document RESF
    resf_match = re.search(r'RESF\s*\d{4}', user_request)
    if resf_match:
        params["document"] = resf_match.group()
    
    # Extraction des figures demandées
    segments = user_request.split()
    best_segment = None
    best_score = 0.0
    
    figure_examples = [
        "croissance du PIB",
        "croissance réelle du PIB",
        "déflateur du PIB",
        "solde public",
        "déficit public"
    ]
    
    for i in range(len(segments)):
        for j in range(i + 1, len(segments) + 1):
            segment = " ".join(segments[i:j])
            segment_embedding = get_embedding(segment)
            
            similarities = []
            for example in figure_examples:
                example_embedding = get_embedding(example)
                similarity = cosine_similarity([segment_embedding], [example_embedding])[0][0]
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            if max_similarity > best_score:
                best_score = max_similarity
                best_segment = segment
    
    if best_score > 0.3:
        params["figures"] = best_segment
    
    # Extraction du type de visualisation
    params["visualization_type"] = find_best_visualization_tool(user_request)
    
    return params

# ============================ Traitement des données ============================
def extract_data_from_text(text, year):
    """
    Extrait les données numériques du texte et les organise en DataFrame
    """
    # Patterns pour trouver les nombres avec leurs labels
    patterns = [
        r'([^:]+):\s*([-+]?\d*\.?\d+)',  # Format "label: valeur"
        r'([^:]+)\s*=\s*([-+]?\d*\.?\d+)',  # Format "label = valeur"
        r'([^:]+)\s*de\s*([-+]?\d*\.?\d+)',  # Format "label de valeur"
        r'([^:]+)\s*à\s*([-+]?\d*\.?\d+)',  # Format "label à valeur"
        r'([^:]+)\s*([-+]?\d*\.?\d+)\s*%',  # Format "label valeur%"
    ]
    
    data = {}
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for label, valeur in matches:
            label = label.strip().lower()
            try:
                valeur = float(valeur)
                if label not in data or abs(valeur) > abs(data[label]):
                    data[label] = valeur
            except ValueError:
                continue
    
    # Création du DataFrame
    df = pd.DataFrame([data])
    df['Annee'] = year
    
    # Sauvegarde dans un fichier Excel
    excel_path = os.path.join(path_tables, f"RESF_{year}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"✅ Données sauvegardées dans {excel_path}")
    
    return df

def process_user_request(user_request):
    """
    Traite la requête de l'utilisateur et génère la réponse appropriée
    """
    try:
        # Analyse de la requête
        params = analyze_user_request(user_request)
        
        if not params["figures"] or not params["document"]:
            return "Je n'ai pas pu extraire toutes les informations nécessaires. Veuillez reformuler votre demande."
        
        # Vérification et traitement OCR si nécessaire
        year = params["year"]
        txt_file = f"RESF_{year}.txt"
        txt_path = os.path.join(path_txt, txt_file)
        
        if not os.path.exists(txt_path):
            print(f"📄 Fichier OCR non trouvé pour {year}, traitement OCR en cours...")
            txt_path = process_ocr(year)
            if txt_path is None:
                return f"❌ Impossible de traiter le document pour {year}"
        
        # Extraction des données
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        df = extract_data_from_text(content, year)
        
        # Sélection de la colonne appropriée
        best_column = find_best_matching_column(params["figures"], df.columns)
        if not best_column:
            return f"Je n'ai pas pu trouver une correspondance précise pour '{params['figures']}'"
        
        # Préparation des données pour la visualisation
        data_series = df[["Annee", best_column]].to_dict('records')
        
        # Création de la visualisation
        if params["visualization_type"] == "create_histogram":
            graph_path = create_histogram(data_series, title=f"{best_column} en {year}")
        else:
            graph_path = create_line_plot(data_series, title=f"Évolution de {best_column} en {year}")
        
        return f"J'ai créé une visualisation des données. Vous pouvez la trouver dans : {graph_path}"
        
    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}"

def main():
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

if __name__ == "__main__":
    main() 