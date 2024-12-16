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
from panda_vision.utils.hash_utils import compute_sha256
from panda_vision.reader.AbsReaderWriter import AbsReaderWriter
from panda_vision.reader.DiskReaderWriter import DiskReaderWriter
from panda_vision.tools.common import PDFParser, DrawingConfig, OutputConfig, ProcessingConfig
from panda_vision.model.doc_analyze_by_custom_model import ModelSingleton

class Environment:
    """Gestion de l'environnement et des configurations"""
    
    LATEX_DELIMITERS = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": '$', "right": '$', "display": False}
    ]
    
    ALL_LANGUAGES = [""] + [
        # Latin languages
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german',
    ]
    
    @staticmethod
    def setup():
        os.system('wget https://raw.githubusercontent.com/EVANE-hub/LayoutFile/refs/heads/main/get_models_script.py -O get_models_script.py')
        os.system('python get_models_script.py')
        os.system("sed -i 's|cpu|cuda|g' /home/user/PANDA-VISION-CONFIG.json")
        os.system('cp -r paddleocr /home/user/.paddleocr')

    @staticmethod
    def init_models():
        try:
            # Création d'une seule instance du singleton
            model_singleton = ModelSingleton()
            return (model_singleton.get_model(False, False) and 
                   model_singleton.get_model(True, False))
        except Exception as e:
            logger.exception(e)
            return False

class PDFProcessor:
    """Traitement des fichiers PDF"""
    
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def read_file(self, path):
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    def parse_pdf(self, doc_path, end_page_id, is_ocr, layout_mode, 
                 formula_enable, table_enable, language):
        try:
            file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
            pdf_data = self.read_file(doc_path)
            parse_method = "ocr" if is_ocr else "auto"
            
            # Création du parser PDF avec la nouvelle classe
            parser = PDFParser(
                pdf_file_name=file_name,
                pdf_bytes=pdf_data,
                model_list=[],
                parse_method=parse_method,
                output_dir=self.output_dir
            )
            
            # Configuration du traitement
            processing_config = ProcessingConfig(
                end_page_id=end_page_id,
                lang=language,
                layout_model=layout_mode,
                formula_enable=formula_enable,
                table_enable=table_enable
            )
            
            # Exécution du parsing
            parser.parse(
                drawing_config=DrawingConfig(),
                output_config=OutputConfig(),
                processing_config=processing_config
            )
            
            return parser.local_md_dir, file_name

        except Exception as e:
            logger.exception(e)
            return None, None

    def process_files(self, file_path, end_pages, is_ocr, layout_mode, 
                     formula_enable, table_enable, language):
        local_md_dir, file_name = self.parse_pdf(
            file_path, end_pages - 1, is_ocr,
            layout_mode, formula_enable, table_enable, language
        )
        
        if not local_md_dir or not file_name:
            return None, None, None, None

        archive_zip_path = os.path.join(
            self.output_dir, compute_sha256(local_md_dir) + ".zip")
        
        self._compress_directory(local_md_dir, archive_zip_path)
        md_content, txt_content = self._process_markdown(local_md_dir, file_name)
        new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")
        
        return md_content, txt_content, archive_zip_path, new_pdf_path

    def _compress_directory(self, directory_path, output_zip_path):
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)

    def _process_markdown(self, local_md_dir, file_name):
        md_path = os.path.join(local_md_dir, file_name + ".md")
        with open(md_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        md_content = self._replace_image_with_base64(txt_content, local_md_dir)
        return md_content, txt_content

    def _replace_image_with_base64(self, markdown_text, image_dir_path):
        pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'
        def replace(match):
            relative_path = match.group(1)
            full_path = os.path.join(image_dir_path, relative_path)
            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"
        return re.sub(pattern, replace, markdown_text)

class GradioInterface:
    """Interface utilisateur Gradio"""
    
    def __init__(self):
        self.processor = PDFProcessor()

    def create_interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(variant='panel', scale=5):
                    file = gr.File(label="Please upload a PDF or image", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
                    max_pages = gr.Slider(1, 10, 5, step=1, label="Max convert pages")
                    
                    with gr.Row():
                        layout_mode = gr.Dropdown(["layoutlmv3", "doclayout_yolo"], label="Layout model", value="doclayout_yolo")
                        language = gr.Dropdown(Environment.ALL_LANGUAGES, label="Language", value="")
                    
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
                            md = gr.Markdown(label="Markdown rendering", height=900,
                                           show_copy_button=True, latex_delimiters=Environment.LATEX_DELIMITERS,
                                           line_breaks=True)
                        with gr.Tab("Markdown text"):
                            md_text = gr.TextArea(lines=45, show_copy_button=True)

            file.upload(fn=self._to_pdf, inputs=file, outputs=pdf_show)
            change_bu.click(
                fn=self.processor.process_files,
                inputs=[pdf_show, max_pages, is_ocr, layout_mode, 
                       formula_enable, table_enable, language],
                outputs=[md, md_text, output_file, pdf_show]
            )
            clear_bu.add([file, md, pdf_show, md_text, output_file, 
                         is_ocr, table_enable, language])

        return demo

    @staticmethod
    def _to_pdf(file_path):
        with pymupdf.open(file_path) as f:
            if f.is_pdf:
                return file_path
            pdf_bytes = f.convert_to_pdf()
            unique_filename = f"{uuid.uuid4()}.pdf"
            tmp_file_path = os.path.join(
                os.path.dirname(file_path), unique_filename)
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)
            return tmp_file_path

if __name__ == "__main__":
    Environment.setup()
    if Environment.init_models():
        interface = GradioInterface()
        demo = interface.create_interface()
        demo.launch(ssr_mode=False)
    else:
        logger.error("Failed to initialize models")