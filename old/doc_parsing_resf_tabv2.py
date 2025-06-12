import os
from mistralai import Mistral
import re
from collections import Counter
from credentials.key import key_Mistral, HF_TOKEN
from huggingface_hub import login
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple
import time
from requests.exceptions import RequestException

# ============================ Configuration ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes

# ============================ Initialisation des clés API ============================
os.environ["HF_TOKEN"] = HF_TOKEN
login(os.environ["HF_TOKEN"])

# ============================ Initialisation du client Mistral ============================
client = Mistral(api_key=key_Mistral)

# ============================ Répertoires ============================
BASE_DIR = Path("C:/Users/menar/Documents/projet_agent_fipu")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
RESF_DIR = INPUT_DIR / "RESF"
TABLES_DIR = OUTPUT_DIR / "tables"

def setup_directories() -> None:
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def process_document(pdf_path: str) -> List[str]:
    """
    Traite un document PDF et extrait les tableaux.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Liste des tableaux extraits
    """
    try:
        logger.info(f"Traitement du document: {pdf_path}")
        
        # Upload du PDF
        uploaded_pdf = client.files.upload(
            file={
                "file_name": pdf_path,
                "content": open(pdf_path, "rb"),
            },
            purpose="ocr"
        )

        # Obtenir l'URL signée
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

        # Effectuer l'OCR avec tentatives de réessai
        ocr_response = None
        for attempt in range(MAX_RETRIES):
            try:
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "document_url",
                        "document_url": signed_url.url,
                    }
                )
                break
            except RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Tentative {attempt + 1} échouée, nouvelle tentative dans {RETRY_DELAY} secondes...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise e

        if ocr_response is None:
            raise Exception("Échec de l'OCR après toutes les tentatives")

        # Extraire le texte OCRisé
        ocr_text = "\n".join(page.markdown for page in ocr_response.pages if page.markdown)
        
        # Log du texte OCRisé pour debug
        logger.debug(f"Texte OCRisé extrait : {ocr_text[:500]}...")  # Affiche les 500 premiers caractères

        # Extraire les tableaux 
        table_pattern = r"(\|[^\n]+\|(?:\n\|[^\n]+\|)+)"
        potential_tables = re.findall(table_pattern, ocr_text)
        
        # Filtrer pour ne garder que les vrais tableaux
        valid_tables = []
        for table in potential_tables:
            lines = table.strip().split("\n")
            # Vérifier que c'est un vrai tableau (au moins 3 lignes et une ligne de séparation)
            if len(lines) >= 3 and any(re.match(r"^\|[-: ]+\|$", line) for line in lines):
                valid_tables.append(table)
        
        return valid_tables
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du document {pdf_path}: {str(e)}")
        return []

def clean_table_structure(table_text: str) -> Tuple[str, int]:
    """
    Corrige les tableaux en ajoutant des colonnes vides aux lignes qui n'ont pas le nombre de colonnes majoritaire.
    
    Args:
        table_text: Texte du tableau en format Markdown
        
    Returns:
        Tuple contenant le tableau corrigé et le nombre de lignes modifiées
    """
    lines = table_text.strip().split("\n")
    col_counts = [line.count("|") for line in lines]
    most_common_cols = Counter(col_counts).most_common(1)[0][0]

    modified_lines = []
    modified_count = 0
    
    for line in lines:
        current_col_count = line.count("|")
        if current_col_count < most_common_cols:
            additional_columns = most_common_cols - current_col_count
            line += " |" * additional_columns
            modified_count += 1
        modified_lines.append(line)

    return "\n".join(modified_lines), modified_count

def table_to_dataframe(table_text: str) -> pd.DataFrame:
    """
    Convertit un tableau Markdown en DataFrame pandas.
    
    Args:
        table_text: Texte du tableau en format Markdown
        
    Returns:
        DataFrame pandas
    """
    lines = table_text.strip().split("\n")
    # Ignorer la ligne de séparation
    lines = [line for line in lines if not re.match(r"^\|[-: ]+\|$", line)]
    
    # Extraire les données
    data = []
    max_cols = 0
    
    # Première passe pour déterminer le nombre maximum de colonnes
    for line in lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        max_cols = max(max_cols, len(cells))
    
    # Deuxième passe pour normaliser toutes les lignes
    for line in lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        # Ajouter des cellules vides si nécessaire
        while len(cells) < max_cols:
            cells.append("")
        # Nettoyer les symboles $ des cellules
        cells = [cell.replace("$", "") for cell in cells]
        data.append(cells)
    
    # Créer le DataFrame
    if data:
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    return pd.DataFrame()

def process_all_documents() -> None:
    """Traite tous les documents PDF dans le répertoire RESF."""
    setup_directories()
    
    # Trouver tous les fichiers PDF
    pdf_files = list(RESF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("Aucun fichier PDF trouvé dans le répertoire RESF")
        return

    for pdf_file in pdf_files:
        logger.info(f"Traitement du fichier: {pdf_file.name}")
        tables = process_document(str(pdf_file))
        
        if not tables:
            logger.warning(f"Aucun tableau trouvé dans {pdf_file.name}")
            continue

        # Créer un fichier Excel pour ce document
        excel_path = TABLES_DIR / f"{pdf_file.stem}_tableaux.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Créer le sommaire
            summary_data = []
            for i, table in enumerate(tables, 1):
                corrected_table, modified_lines = clean_table_structure(table)
                df = table_to_dataframe(corrected_table)
                if not df.empty:
                    # Sauvegarder le tableau dans un onglet
                    sheet_name = f"Tableau_{i}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Ajouter les informations au sommaire
                    summary_data.append({
                        'Numéro du tableau': i,
                        'Nom de l\'onglet': sheet_name,
                        'Nombre de lignes': len(df),
                        'Nombre de colonnes': len(df.columns),
                        'Lignes modifiées': modified_lines
                    })
            
            # Créer et sauvegarder le sommaire
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Sommaire', index=False)
                logger.info(f"✅ Export Excel terminé pour {pdf_file.name} : {excel_path}")
            else:
                logger.warning(f"Aucun tableau n'a été extrait de {pdf_file.name}")

if __name__ == "__main__":
    process_all_documents()
