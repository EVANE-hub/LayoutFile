# Imports standards
import base64
import os
import re
import time
import uuid
import zipfile
from pathlib import Path

# Imports tiers
import gradio as gr
import pymupdf
from gradio_pdf import PDF
from loguru import logger

# Imports locaux
from panda_vision.libs.hash_utils import compute_sha256
from panda_vision.rw.AbsReaderWriter import AbsReaderWriter
from panda_vision.rw.DiskReaderWriter import DiskReaderWriter
from panda_vision.tools.common import do_parse, prepare_env

# Configuration initiale
os.system('wget https://raw.githubusercontent.com/EVANE-hub/LayoutFile/refs/heads/main/get_models_script.py -O get_models_script.py')
os.system('python get_models_script.py')
os.system("sed -i 's|cpu|cuda|g' /home/user/PANDA-VISION-CONFIG.json")
os.system('cp -r paddleocr /home/user/.paddleocr')

# Configuration LaTeX
latex_delimiters = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": '$', "right": '$', "display": False}
]

# Configuration des langues
latin_lang = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']

all_lang = [""]
all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])

def read_fn(path):
    """Lecture d'un fichier avec DiskReaderWriter."""
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    """Parse un fichier PDF et génère les fichiers de sortie."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)
        parse_method = "ocr" if is_ocr else "auto"
        
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir, file_name, pdf_data, [], parse_method, False,
            end_page_id=end_page_id, layout_model=layout_mode,
            formula_enable=formula_enable, table_enable=table_enable,
            lang=language
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)

def compress_directory_to_zip(directory_path, output_zip_path):
    """Compresse un répertoire en fichier ZIP."""
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1

def image_to_base64(image_path):
    """Convertit une image en base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def replace_image_with_base64(markdown_text, image_dir_path):
    """Remplace les liens d'images par leur équivalent base64."""
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'
    
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"
    
    return re.sub(pattern, replace, markdown_text)

def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
    """Convertit un fichier en Markdown."""
    local_md_dir, file_name = parse_pdf(
        file_path, './output', end_pages - 1, is_ocr,
        layout_mode, formula_enable, table_enable, language
    )
    
    # Création de l'archive
    archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    logger.info("Compression réussie" if zip_archive_success == 0 else "Compression échouée")
    
    # Lecture et conversion du contenu Markdown
    md_path = os.path.join(local_md_dir, file_name + ".md")
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    
    new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")
    return md_content, txt_content, archive_zip_path, new_pdf_path

def init_model():
    """Initialise les modèles nécessaires."""
    from panda_vision.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)
        logger.info("txt_model init final")
        ocr_model = model_manager.get_model(True, False)
        logger.info("ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1

def to_pdf(file_path):
    """Convertit un fichier en PDF si nécessaire."""
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        
        pdf_bytes = f.convert_to_pdf()
        unique_filename = f"{uuid.uuid4()}.pdf"
        tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)
        
        with open(tmp_file_path, 'wb') as tmp_pdf_file:
            tmp_pdf_file.write(pdf_bytes)
        return tmp_file_path

# Initialisation du modèle
model_init = init_model()
logger.info(f"model_init: {model_init}")

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                file = gr.File(label="Please upload a PDF or image", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
                max_pages = gr.Slider(1, 10, 5, step=1, label="Max convert pages")
                
                with gr.Row():
                    layout_mode = gr.Dropdown(["layoutlmv3", "doclayout_yolo"], label="Layout model", value="layoutlmv3")
                    language = gr.Dropdown(all_lang, label="Language", value="")
                
                with gr.Row():
                    formula_enable = gr.Checkbox(label="Enable formula recognition", value=True)
                    is_ocr = gr.Checkbox(label="Force enable OCR", value=False)
                    table_enable = gr.Checkbox(label="Enable table recognition(test)", value=False)
                
                with gr.Row():
                    change_bu = gr.Button("Convert")
                    clear_bu = gr.ClearButton(value="Clear")
                
                pdf_show = PDF(label="PDF preview", interactive=True, height=800)

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label="convert result", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Markdown rendering"):
                        md = gr.Markdown(
                            label="Markdown rendering", height=900,
                            show_copy_button=True, latex_delimiters=latex_delimiters,
                            line_breaks=True
                        )
                    with gr.Tab("Markdown text"):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)

        # Configuration des événements
        file.upload(fn=to_pdf, inputs=file, outputs=pdf_show)
        change_bu.click(
            fn=to_markdown,
            inputs=[pdf_show, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
            outputs=[md, md_text, output_file, pdf_show],
            api_name=False
        )
        clear_bu.add([file, md, pdf_show, md_text, output_file, is_ocr, table_enable, language])

    demo.launch(ssr_mode=False)
