import os
from mistralai import Mistral
import re
from collections import Counter
from credentials.key import key_Mistral
from parametres.params import parent_dir, input_dir, output_dir, tables_dir, txt_dir, report_dir, RESF_dir
import pandas as pd
import json


# Initialiser le client Mistral
client = Mistral(api_key=key_Mistral)


path = parent_dir + "/" + input_dir + "/" + RESF_dir
path_output = parent_dir + "/" + output_dir + "/" + tables_dir

# Charger le fichier PDF
pdf_path = path + '/RESF_2020.pdf'

uploaded_pdf = client.files.upload(
    file={
        "file_name": pdf_path,
        "content": open(pdf_path, "rb"),
    },
    purpose="ocr"
)  

signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

# Effectuer l'OCR
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    }
)

# Extraire le texte OCRisé
ocr_text = "\n".join(page.markdown for page in ocr_response.pages if page.markdown)

# Regex pour extraire les tableaux Markdown
table_pattern = r"(\|(?:[^\n]+\|)+\n(?:\|(?:[-: ]+\|)+\n)+(?:\|(?:[^\n]+\|)+\n)*)"
tables = re.findall(table_pattern, ocr_text)

# Créer un dossier pour sauvegarder les tableaux
output_dir = path_output + "/tableaux_extraits"
os.makedirs(output_dir, exist_ok=True)

def clean_table_structure(table_text):
    """
    Corrige les tableaux en ajoutant des colonnes vides aux lignes qui n'ont pas le nombre de colonnes majoritaire.
    Retourne le tableau corrigé et le nombre de lignes modifiées.
    """
    lines = table_text.strip().split("\n")
    
    # Compter le nombre de colonnes pour chaque ligne
    col_counts = [line.count("|") for line in lines]
    
    # Trouver le nombre de colonnes le plus fréquent
    most_common_cols = Counter(col_counts).most_common(1)[0][0]

    # Ajouter des colonnes vides aux lignes avec moins de colonnes que le nombre majoritaire
    modified_lines = []
    modified_count = 0
    
    for line in lines:
        current_col_count = line.count("|")
        if current_col_count < most_common_cols:
            # Ajouter des colonnes vides (représentées par "| |")
            additional_columns = most_common_cols - current_col_count
            line += " |" * additional_columns
            modified_count += 1
        modified_lines.append(line)

    return "\n".join(modified_lines), modified_count

def contains_numbers(table_text):
    """
    Vérifie si un tableau contient des chiffres.
    Retourne True si le tableau contient au moins un chiffre.
    """
    return bool(re.search(r'\d', table_text))

def is_time_series_table(table_text):
    """
    Vérifie si le tableau contient une série temporelle.
    Retourne True si le tableau contient des années ou des périodes.
    """
    # Recherche de patterns d'années (4 chiffres) ou de périodes
    year_pattern = r'\b(19|20)\d{2}\b'
    period_pattern = r'\b(T[1-4]|S[1-2]|Q[1-4])\b'
    
    return bool(re.search(year_pattern, table_text) or re.search(period_pattern, table_text))

def extract_time_series(table_text):
    """
    Extrait les séries temporelles d'un tableau.
    Retourne un dictionnaire avec les séries et leurs valeurs.
    """
    lines = table_text.strip().split('\n')
    if len(lines) < 2:
        return None
    
    # Extraire les en-têtes
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    if not headers:
        return None
    
    # Trouver l'index de la colonne temporelle (années ou périodes)
    time_col_idx = None
    for i, header in enumerate(headers):
        if re.search(r'\b(19|20)\d{2}\b', header) or re.search(r'\b(T[1-4]|S[1-2]|Q[1-4])\b', header):
            time_col_idx = i
            break
    
    if time_col_idx is None:
        return None
    
    # Extraire les données
    series_data = {}
    for line in lines[2:]:  # Ignorer l'en-tête et la ligne de séparation
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        
        # Vérifier que la ligne a assez de colonnes
        if len(cells) <= time_col_idx or len(cells) != len(headers):
            continue
            
        time_value = cells[time_col_idx]
        if not re.search(r'\d', time_value):  # Ignorer les lignes sans chiffres
            continue
            
        # Créer une entrée pour chaque colonne (sauf la colonne temporelle)
        for i, value in enumerate(cells):
            if i != time_col_idx and i < len(headers) and value.strip():
                series_name = f"{headers[i]}_{time_value}"
                try:
                    # Nettoyer la valeur (enlever les symboles $, espaces, etc.)
                    clean_value = float(re.sub(r'[^\d.-]', '', value))
                    series_data[series_name] = clean_value
                except ValueError:
                    continue
    
    return series_data if series_data else None

# Sauvegarde des tableaux après correction
table_count = 0
all_series = {}

for i, table in enumerate(tables, 1):
    # Vérifier si le tableau contient des chiffres
    if not contains_numbers(table):
        print(f"⏭️ Tableau {i} ignoré (pas de chiffres)")
        continue
        
    table_count += 1
    corrected_table, modified_lines = clean_table_structure(table)
    
    # Enregistrer le tableau corrigé
    file_path = os.path.join(output_dir, f"tableau_{table_count}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(corrected_table)

    # Extraire les séries temporelles si c'est un tableau de séries
    if is_time_series_table(corrected_table):
        series_data = extract_time_series(corrected_table)
        if series_data:
            all_series.update(series_data)
            print(f"📊 Tableau {table_count} : {len(series_data)} séries extraites")
        else:
            print(f"ℹ️ Tableau {table_count} : pas de séries temporelles valides")
    else:
        print(f"ℹ️ Tableau {table_count} : pas un tableau de séries temporelles")

    if modified_lines > 0:
        print(f"🟡 Tableau {table_count} corrigé et enregistré : {modified_lines} ligne(s) modifiée(s)")
    else:
        print(f"✅ Tableau {table_count} enregistré sans modification")

# Sauvegarder toutes les séries dans un fichier JSON
if all_series:
    series_file = os.path.join(output_dir, "series_temporelles.json")
    with open(series_file, "w", encoding="utf-8") as f:
        json.dump(all_series, f, indent=2, ensure_ascii=False)
    print(f"\n💾 {len(all_series)} séries temporelles sauvegardées dans : '{series_file}'")

print(f"\n🔍 Vérification et correction terminées. {table_count} tableaux traités dans : '{output_dir}'.")

# Chemin du fichier JSON
json_path = os.path.join(output_dir, "series_temporelles.json")
excel_path = os.path.join(output_dir, "series_temporelles.xlsx")

# Charger les données JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Conversion en DataFrame
df = pd.DataFrame(list(data.items()), columns=["Série", "Valeur"])

# Exporter vers Excel
df.to_excel(excel_path, index=False)
print(f"✅ Export Excel terminé : {excel_path}")