# PANDA-VISION

Un outil puissant pour convertir des documents PDF en Markdown avec une prise en charge avancée de la mise en page, des formules mathématiques, des tableaux et de l'OCR.

## Fonctionnalités

- Extraction de texte par OCR ou méthode directe (TXT)
- Reconnaissance de la mise en page avec LayoutLMv3
- Détection et extraction des formules mathématiques 
- Reconnaissance des tableaux
- Support multilingue
- Interface en ligne de commande et interface web Gradio
- Prise en charge du stockage local et S3

## Installation

```bash
pip install -r requirements.txt

pip install -e .[detectron2]
```

## Configuration

1. Créez un fichier `PANDA-VISION-CONFIG.json` dans votre répertoire utilisateur :

```json
{
  "models-dir": "/chemin/vers/modeles",
  "device-mode": "cuda",  // ou "cpu"
  "layout-config": {
    "model": "layoutlmv3"
  }
}
```

2. Téléchargez les modèles requis :
```bash
python get_models_script.py
```

## Utilisation

### En Python

```python
from panda_vision.data.data_reader_writer import FileBasedDataWriter
from panda_vision.pipe.OCRPipe import OCRPipe

# Configurer les chemins
image_writer = FileBasedDataWriter("./output/images")
pdf_bytes = open("document.pdf", "rb").read()

# Initialiser et exécuter le pipeline
pipe = OCRPipe(pdf_bytes, [], image_writer)
pipe.pipe_classify()
pipe.pipe_analyze()
pipe.pipe_parse()
pipe.pipe_mk_markdown("./output/images")
```

### Interface Web

```bash
python run_in_gradio.py
```

## Structure du Projet

- `panda_vision/`
  - `model/` : Implémentations des modèles (LayoutLMv3, OCR, etc.)
  - `pipe/` : Pipeline de traitement PDF
  - `pre_proc/` : Prétraitement des documents
  - `utils/` : Utilitaires communs
  - `data/` : Gestion des données et E/S
  - `config/` : Configuration du projet

## Langues Supportées

Le système prend en charge de nombreuses langues, notamment :
- Langues latines (français, anglais, allemand, etc.)
- Langues cyrilliques (russe, ukrainien, etc.)
- Langues arabes
