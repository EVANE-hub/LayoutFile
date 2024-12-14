import copy
import json as json_parse
import os
from dataclasses import dataclass
from typing import Optional, List

import fitz
from loguru import logger
import click

from panda_vision.config.make_content_config import DropMode, MakeMode
from panda_vision.data.data_reader_writer import FileBasedDataWriter
from panda_vision.utils.draw_bbox import (draw_layout_bbox, draw_line_sort_bbox, draw_model_bbox, draw_span_bbox)
from panda_vision.pipe.PDFProcessor import PDFProcessor

@dataclass
class PDFConfig:
    """Configuration pour le traitement PDF"""
    output_dir: str
    pdf_file_name: str
    pdf_bytes: bytes
    model_list: List
    parse_method: str
    debug_able: bool
    start_page_id: int = 0
    end_page_id: Optional[int] = None
    lang: Optional[str] = None
    layout_model: Optional[str] = None
    formula_enable: Optional[bool] = None
    table_enable: Optional[bool] = None

@dataclass 
class DrawConfig:
    """Configuration pour le dessin des boîtes"""
    draw_span_bbox: bool = True
    draw_layout_bbox: bool = True
    draw_model_bbox: bool = False
    draw_line_sort_bbox: bool = False

@dataclass
class OutputConfig:
    """Configuration pour la sortie des fichiers"""
    dump_md: bool = True
    dump_middle_json: bool = True
    dump_model_json: bool = True
    dump_orig_pdf: bool = True
    dump_content_list: bool = True
    make_md_mode: MakeMode = MakeMode.MM_MD

@dataclass
class Paths:
    local_image_dir: str
    local_md_dir: str

class PDFHandler:
    def __init__(self, pdf_config: PDFConfig, draw_config: DrawConfig, output_config: OutputConfig):
        self.pdf_config = pdf_config
        self.draw_config = draw_config
        self.output_config = output_config
        self._setup_debug_mode()
        
    def _setup_debug_mode(self):
        """Configure le mode debug si activé"""
        if self.pdf_config.debug_able:
            logger.warning('debug mode is on')
            self.draw_config.draw_model_bbox = True
            self.draw_config.draw_line_sort_bbox = True

    def _prepare_env(self) -> tuple:
        """Prépare l'environnement de sortie"""
        local_parent_dir = os.path.join(
            self.pdf_config.output_dir,
            self.pdf_config.pdf_file_name,
            self.pdf_config.parse_method
        )
        local_image_dir = os.path.join(str(local_parent_dir), 'images')
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_parent_dir, exist_ok=True)
        return local_image_dir, local_parent_dir

    def _convert_pdf_bytes(self) -> bytes:
        """Convertit les bytes PDF avec pymupdf"""
        document = fitz.open('pdf', self.pdf_config.pdf_bytes)
        output_document = fitz.open()
        end_page = (self.pdf_config.end_page_id if self.pdf_config.end_page_id is not None 
                   and self.pdf_config.end_page_id >= 0 else len(document) - 1)
        
        if end_page > len(document) - 1:
            logger.warning('end_page_id is out of range, use pdf_docs length')
            end_page = len(document) - 1
            
        output_document.insert_pdf(document, 
                                 from_page=self.pdf_config.start_page_id,
                                 to_page=end_page)
        return output_document.tobytes()

    def _process_pdf(self, image_writer, local_md_dir: str):
        """Traite le PDF avec le pipeline"""
        pipe = PDFProcessor(
            self.pdf_config.pdf_bytes,
            self.pdf_config.model_list,
            image_writer,
            pdf_type=self.pdf_config.parse_method,
            is_debug=True,
            lang=self.pdf_config.lang,
            layout_model=self.pdf_config.layout_model,
            formula_enable=self.pdf_config.formula_enable,
            table_enable=self.pdf_config.table_enable
        )
        pipe.process()
        return pipe

    def process(self):
        """Point d'entrée principal pour le traitement"""
        if self.pdf_config.lang == "":
            self.pdf_config.lang = None

        self.pdf_config.pdf_bytes = self._convert_pdf_bytes()
        orig_model_list = copy.deepcopy(self.pdf_config.model_list)
        paths = Paths(self._prepare_env())
        image_writer = FileBasedDataWriter(paths.local_image_dir)
        md_writer = FileBasedDataWriter(paths.local_md_dir)
        image_dir = str(os.path.basename(paths.local_image_dir))

        pipe = self._process_pdf(image_writer, paths.local_md_dir)
        self._handle_drawing(pipe, orig_model_list, paths.local_md_dir)
        self._handle_output(pipe, md_writer, image_dir, orig_model_list)
        return paths
        
    def _handle_drawing(self, pipe, orig_model_list: list, local_md_dir: str):
        """Gère le dessin des boîtes"""
        pdf_info = pipe.pdf_mid_data['pdf_info']
        
        if self.draw_config.draw_layout_bbox:
            draw_layout_bbox(pdf_info, self.pdf_config.pdf_bytes, local_md_dir, self.pdf_config.pdf_file_name)
        if self.draw_config.draw_span_bbox:
            draw_span_bbox(pdf_info, self.pdf_config.pdf_bytes, local_md_dir, self.pdf_config.pdf_file_name)
        if self.draw_config.draw_model_bbox:
            draw_model_bbox(copy.deepcopy(orig_model_list), self.pdf_config.pdf_bytes, local_md_dir, self.pdf_config.pdf_file_name)
        if self.draw_config.draw_line_sort_bbox:
            draw_line_sort_bbox(pdf_info, self.pdf_config.pdf_bytes, local_md_dir, self.pdf_config.pdf_file_name)

    def _handle_output(self, pipe, md_writer, image_dir: str, orig_model_list: list):
        """Gère la sortie des fichiers"""
        if self.output_config.dump_md:
            md_content = pipe.get_markdown(image_dir, drop_mode=DropMode.NONE, 
                                        make_mode=self.output_config.make_md_mode)
            md_writer.write_string(f'{self.pdf_config.pdf_file_name}.md', md_content)

        if self.output_config.dump_middle_json:
            md_writer.write_string(
                f'{self.pdf_config.pdf_file_name}_middle.json',
                json_parse.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4)
            )

        if self.output_config.dump_model_json:
            md_writer.write_string(
                f'{self.pdf_config.pdf_file_name}_model.json',
                json_parse.dumps(orig_model_list, ensure_ascii=False, indent=4)
            )

        if self.output_config.dump_orig_pdf:
            md_writer.write(
                f'{self.pdf_config.pdf_file_name}_origin.pdf',
                self.pdf_config.pdf_bytes,
            )

        if self.output_config.dump_content_list:
            content_list = pipe.get_content(image_dir, drop_mode=DropMode.NONE)
            md_writer.write_string(
                f'{self.pdf_config.pdf_file_name}_content_list.json',
                json_parse.dumps(content_list, ensure_ascii=False, indent=4)
            )

def entrypoint(output_dir, pdf_file_name, pdf_bytes, model_list, parse_method, 
            debug_able, start_page_id=0, end_page_id=None, lang=None, 
            layout_model=None, formula_enable=None, table_enable=None):
    """Fonction principale pour le parsing de PDF"""
    pdf_config = PDFConfig(
        output_dir=output_dir,
        pdf_file_name=pdf_file_name,
        pdf_bytes=pdf_bytes,
        model_list=model_list,
        parse_method=parse_method,
        debug_able=debug_able,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        lang=lang,
        layout_model=layout_model,
        formula_enable=formula_enable,
        table_enable=table_enable
    )
    
    draw_config = DrawConfig(
        draw_span_bbox=True,
        draw_layout_bbox=True,
        draw_model_bbox=debug_able,
        draw_line_sort_bbox=debug_able
    )
    
    output_config = OutputConfig(
        dump_md=True,
        dump_middle_json=True,
        dump_model_json=True,
        dump_orig_pdf=True,
        dump_content_list=True,
        make_md_mode=MakeMode.MM_MD
    )
    
    handler = PDFHandler(pdf_config, draw_config, output_config)

    return handler.process()

