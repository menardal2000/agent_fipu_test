import os
import time
import torch
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import re
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

# ============================ Modèle d'embeddings local ============================
model = AutoModel.from_pretrained("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

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

# ============================ Résumé par chapitre ============================
path_input = os.path.join(parent_dir, input_dir)
path_resf = os.path.join(path_input, RESF_dir)
path_output = os.path.join(parent_dir, output_dir)
path_graph = os.path.join(path_output, graph_dir)
path_database = os.path.join(path_output, database_dir)
path_txt = os.path.join(path_output, txt_dir)
path_report = os.path.join(path_output, report_dir)

os.makedirs(path_input, exist_ok=True)
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_graph, exist_ok=True)
os.makedirs(path_database, exist_ok=True)
os.makedirs(path_txt, exist_ok=True)
os.makedirs(path_report, exist_ok=True)

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
    
def reponse_question(question, filename):
    print(f"\n📄 Traitement du fichier: {filename}")
    file_path = os.path.join(path_txt, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    document = Document(page_content=content, metadata={"source": filename})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    chunks = text_splitter.split_documents([document])
    
    chroma_path = f"{path_database}/db_{filename.replace('.txt', '')}"
    print("🔍 Initialisation des embeddings Mistral...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=key_Mistral)

    if not os.path.exists(chroma_path):
        print("📚 Création de la base de données Chroma...")
        db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    else:
        print("📚 Chargement de la base de données Chroma existante...")
        db_chroma = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    
    print("🔍 Recherche de documents similaires...")
    docs_chroma = db_chroma.similarity_search_with_score(question, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    prompt = prompt_template.format(context=context_text, question=question)
    print("🤖 Génération de la réponse avec Mistral...")
    response_text = invoke_with_retry(model_chat, prompt)

    # Création du nom du fichier de rapport basé sur le fichier source
    report_filename = f"rapport_{filename.replace('.txt', '.txt')}"
    report_file_path = os.path.join(path_report, report_filename)
    
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(response_text.content)

    print(f"✅ Réponse générée pour {filename} → {report_file_path}")
    return response_text.content

# ============================ Requête avec gestion des erreurs ============================
def invoke_with_retry(model, prompt, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            print(f"📤 Envoi de la requête à l'API Mistral (tentative {retries + 1})...")
            response = model.invoke(prompt)
            print("✅ Réponse reçue de l'API Mistral")
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** retries
                print(f"⚠️ Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"❌ Erreur HTTP: {e}")
                raise e
    raise Exception("❌ Max retries reached, request failed.")

def extraire_numeriques(texte):
    """
    Extrait les valeurs numériques et leurs labels du texte
    """
    # Pattern pour trouver les nombres avec leurs labels
    pattern = r'([^:]+):\s*([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, texte)
    
    resultats = {}
    for label, valeur in matches:
        label = label.strip().lower()
        try:
            valeur = float(valeur)
            resultats[label] = valeur
        except ValueError:
            continue
    return resultats

def generer_graphiques(reponses, question):
    """
    Génère des graphiques à partir des réponses pour chaque année
    """
    # Créer un DataFrame pour stocker les données
    data = {}
    
    for annee, reponse in reponses.items():
        valeurs = extraire_numeriques(reponse)
        if valeurs:
            data[annee] = valeurs
    
    if not data:
        print("⚠️ Aucune donnée numérique trouvée pour générer des graphiques")
        return
    
    # Créer un DataFrame
    df = pd.DataFrame(data).T
    
    # Créer un dossier pour les graphiques s'il n'existe pas
    os.makedirs(path_graph, exist_ok=True)
    
    # Générer différents types de graphiques
    # 1. Graphique en barres pour chaque indicateur
    for colonne in df.columns:
        plt.figure(figsize=(10, 6))
        df[colonne].plot(kind='bar')
        plt.title(f'Évolution de {colonne} par année')
        plt.xlabel('Année')
        plt.ylabel('Valeur')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        graph_filename = f"{colonne.replace(' ', '_')}_evolution.png"
        graph_path = os.path.join(path_graph, graph_filename)
        plt.savefig(graph_path)
        plt.close()
        print(f"✅ Graphique sauvegardé : {graph_path}")
    
    # 2. Graphique en ligne pour tous les indicateurs
    plt.figure(figsize=(12, 6))
    for colonne in df.columns:
        plt.plot(df.index, df[colonne], marker='o', label=colonne)
    
    plt.title('Évolution des indicateurs par année')
    plt.xlabel('Année')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Sauvegarder le graphique combiné
    graph_filename = "evolution_globale.png"
    graph_path = os.path.join(path_graph, graph_filename)
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Graphique global sauvegardé : {graph_path}")

def comparer_reponses(question, annees, generer_graphiques_option=False):
    """
    Compare les réponses pour plusieurs années et génère un rapport comparatif
    """
    print(f"\n🔄 Comparaison des réponses pour les années {annees}")
    
    # Collecter les réponses pour chaque année
    reponses = {}
    for annee in annees:
        filename = f"RESF_{annee}.txt"
        if os.path.exists(os.path.join(path_txt, filename)):
            reponse = reponse_question(question, filename)
            reponses[annee] = reponse
        else:
            print(f"⚠️ Fichier {filename} non trouvé")
    
    # Générer un rapport comparatif
    if reponses:
        rapport_comparatif = f"Rapport comparatif pour la question : {question}\n\n"
        for annee, reponse in reponses.items():
            rapport_comparatif += f"\n=== Année {annee} ===\n{reponse}\n"
        
        # Sauvegarder le rapport comparatif
        rapport_filename = f"rapport_comparatif_{'-'.join(annees)}.txt"
        rapport_path = os.path.join(path_report, rapport_filename)
        with open(rapport_path, "w", encoding="utf-8") as f:
            f.write(rapport_comparatif)
        print(f"✅ Rapport comparatif généré → {rapport_path}")
        
        # Générer les graphiques si demandé
        if generer_graphiques_option:
            generer_graphiques(reponses, question)

def main():
    annees = ["2020", "2021", "2022"]
    question = "Donne moi les chiffres d'inflation, les chiffres de déficit et de croissance de la France en pourcentage du PIB dans les RESF 2020 et 2021 pour toutes les années disponibles"
    
    for annee in annees:
        print(f"\nTraitement de l'année {annee}...")
        
        # Étape 1: OCR
        input_path = process_ocr(annee)
        if input_path is None:
            continue
    
    # Étape 2: Génération de l'analyse comparative
    print("\nVoulez-vous générer des graphiques pour visualiser les données ? (oui/non)")
    reponse = input().lower().strip()
    generer_graphiques_option = reponse in ['oui', 'o', 'yes', 'y']
    
    comparer_reponses(question, annees, generer_graphiques_option)

if __name__ == "__main__":
    main()