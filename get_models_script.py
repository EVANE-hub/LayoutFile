import json
import os
import requests
from huggingface_hub import snapshot_download

PANDA_VISION_PATTERNS = [
    "models/Layout/LayoutLMv3/*",
    "models/Layout/YOLO/*", 
    "models/MFD/YOLO/*",
    "models/MFR/unimernet_small/*", 
    "models/TabRec/TableMaster/*",
    "models/TabRec/StructEqTable/*",
]

LAYOUTREADER_PATTERNS = [
    "*.json",
    "*.safetensors",
]

CONFIG_URL = 'https://github.com/opendatalab/panda_vision/raw/master/magic-pdf.template.json'
CONFIG_FILENAME = 'magic-pdf.json'


def download_json(url):
    """Télécharge et retourne le contenu d'un fichier JSON depuis une URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    """Télécharge, modifie et sauvegarde un fichier JSON avec les modifications spécifiées."""
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        if data.get('config_version', '0.0.0') < '1.0.0':
            data = download_json(url)
    else:
        data = download_json(url)
    data.update(modifications)
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Téléchargement des modèles
    model_dir = snapshot_download(
        'opendatalab/PDF-Extract-Kit-1.0', 
        allow_patterns=PANDA_VISION_PATTERNS
    )
    model_dir = os.path.join(model_dir, 'models')
    layoutreader_model_dir = snapshot_download(
        'hantian/layoutreader', 
        allow_patterns=LAYOUTREADER_PATTERNS
    )

    print(f'Le répertoire des modèles est : {model_dir}')
    print(f'Le répertoire du modèle layoutreader est : {layoutreader_model_dir}')

    # Configuration du fichier JSON
    config_file = os.path.join(os.path.expanduser('~'), CONFIG_FILENAME)
    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(CONFIG_URL, config_file, json_mods)
    print(f'Le fichier de configuration a été configuré avec succès, le chemin est : {config_file}')
