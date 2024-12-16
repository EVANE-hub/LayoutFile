import json
import os
import requests
from huggingface_hub import snapshot_download
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelPaths:
    PANDA_VISION_PATTERNS = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*", 
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small/*", 
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    LAYOUTREADER_PATTERNS = ["*.json", "*.safetensors"]

class ModelDownloader:
    def __init__(self, config_url: str, config_filename: str):
        self.config_url = config_url
        self.config_filename = config_filename
        self.config_file = os.path.join(os.path.expanduser('~'), config_filename)

    def _download_json(self) -> Dict[str, Any]:
        """Télécharge et retourne le contenu d'un fichier JSON depuis une URL."""
        try:
            response = requests.get(self.config_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Erreur lors du téléchargement du JSON: {e}")

    def _update_config(self, model_dir: str, layoutreader_model_dir: str) -> None:
        """Met à jour le fichier de configuration avec les nouveaux chemins."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file) as f:
                    data = json.load(f)
                if data.get('config_version', '0.0.0') < '1.0.0':
                    data = self._download_json()
            else:
                data = self._download_json()

            data.update({
                'models-dir': model_dir,
                'layoutreader-model-dir': layoutreader_model_dir
            })

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise Exception(f"Erreur lors de la mise à jour de la configuration: {e}")

    def download_models(self) -> None:
        """Télécharge les modèles et met à jour la configuration."""
        try:
            model_dir = snapshot_download(
                'Leyogho/LayoutVision', 
                allow_patterns=ModelPaths.PANDA_VISION_PATTERNS
            )
            model_dir = os.path.join(model_dir, 'models')

            layoutreader_model_dir = snapshot_download(
                'hantian/layoutreader', 
                allow_patterns=ModelPaths.LAYOUTREADER_PATTERNS
            )

            self._update_config(model_dir, layoutreader_model_dir)
            
            print(f'Le répertoire des modèles est : {model_dir}')
            print(f'Le répertoire du modèle layoutreader est : {layoutreader_model_dir}')
            print(f'Le fichier de configuration a été configuré avec succès, le chemin est : {self.config_file}')
            
        except Exception as e:
            raise Exception(f"Erreur lors du téléchargement des modèles: {e}")

def main():
    try:
        downloader = ModelDownloader(
            config_url='https://raw.githubusercontent.com/EVANE-hub/LayoutFile/refs/heads/main/PANDA-VISION-CONFIG.json',
            config_filename='PANDA-VISION-CONFIG.json'
        )
        downloader.download_models()
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == '__main__':
    main()
