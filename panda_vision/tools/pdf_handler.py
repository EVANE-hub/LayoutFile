import copy
import json as json_parse
import os
import fitz
from loguru import logger
import panda_vision.model as model_config
from panda_vision.config.make_content_config import DropMode, MakeMode
from panda_vision.data.data_reader_writer import FileBasedDataWriter
from panda_vision.utils.draw_bbox import (draw_layout_bbox, draw_line_sort_bbox, 
                                        draw_model_bbox, draw_span_bbox)
from panda_vision.pipe.PDFProcessor import PDFProcessor
from dataclasses import dataclass
from typing import Optional
from panda_vision.pipe.PDFProcessor import PDFType

@dataclass
class DrawingConfig:
    draw_span_bbox: bool = True
    draw_layout_bbox: bool = True
    draw_model_bbox: bool = False
    draw_line_sort_bbox: bool = False

@dataclass
class OutputConfig:
    dump_md: bool = True
    dump_middle_json: bool = True
    dump_model_json: bool = True
    dump_orig_pdf: bool = True
    dump_content_list: bool = True
    make_md_mode: MakeMode = MakeMode.MM_MD

@dataclass
class ProcessingConfig:
    debug_able: bool = False
    start_page_id: int = 0
    end_page_id: Optional[int] = None
    lang: Optional[str] = None
    layout_model: Optional[str] = None
    formula_enable: Optional[bool] = None
    table_enable: Optional[bool] = None

class PDFParser:
    VALID_PARSE_METHODS = ['ocr', 'txt', 'auto']
    def __init__(self, pdf_file_name, pdf_bytes, model_list, parse_method=PDFType.UNION , output_dir='./output'):
        if parse_method not in self.VALID_PARSE_METHODS:
            raise ValueError(f"parse_method must be one of {self.VALID_PARSE_METHODS}")
        self.output_dir = output_dir
        self.parse_method = parse_method 
        self.pdf_file_name = pdf_file_name
        self.pdf_bytes = pdf_bytes
        self.model_list = model_list
        self.md_writer = None
        
    def prepare_env(self):
        local_parent_dir = os.path.join(self.output_dir, self.pdf_file_name, self.parse_method)
        self.local_image_dir = os.path.join(str(local_parent_dir), 'images')
        self.local_md_dir = local_parent_dir
        os.makedirs(self.local_image_dir, exist_ok=True)
        os.makedirs(self.local_md_dir, exist_ok=True)
        self.image_writer = FileBasedDataWriter(self.local_image_dir)
        self.md_writer = FileBasedDataWriter(self.local_md_dir)
        
    def convert_pdf_bytes(self, start_page_id=0, end_page_id=None):
        document = fitz.open('pdf', self.pdf_bytes)
        output_document = fitz.open()
        end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(document) - 1
        if end_page_id > len(document) - 1:
            logger.warning('end_page_id is out of range, use pdf_docs length')
            end_page_id = len(document) - 1
        output_document.insert_pdf(document, from_page=start_page_id, to_page=end_page_id)
        self.pdf_bytes = output_document.tobytes()

    def _setup_debug_mode(self, debug_able, f_draw_model_bbox, f_draw_line_sort_bbox):
        if debug_able:
            logger.warning('debug mode is on')
            return True, True
        return f_draw_model_bbox, f_draw_line_sort_bbox

    def _process_pdf(self, lang, layout_model, formula_enable, table_enable):
        pipe = PDFProcessor(
            self.pdf_bytes, 
            self.model_list, 
            self.image_writer,
            pdf_type=self.parse_method,
            is_debug=True,
            lang=lang if lang != "" else None,
            layout_model=layout_model,
            formula_enable=formula_enable,
            table_enable=table_enable
        )
        pipe.process()
        return pipe

    def _handle_drawings(self, pdf_info, orig_model_list, drawing_flags):
        f_draw_layout_bbox, f_draw_span_bbox, f_draw_model_bbox, f_draw_line_sort_bbox = drawing_flags
        
        if f_draw_layout_bbox:
            draw_layout_bbox(pdf_info, self.pdf_bytes, self.local_md_dir, self.pdf_file_name)
        if f_draw_span_bbox:
            draw_span_bbox(pdf_info, self.pdf_bytes, self.local_md_dir, self.pdf_file_name)
        if f_draw_model_bbox:
            draw_model_bbox(copy.deepcopy(orig_model_list), self.pdf_bytes, self.local_md_dir, self.pdf_file_name)
        if f_draw_line_sort_bbox:
            draw_line_sort_bbox(pdf_info, self.pdf_bytes, self.local_md_dir, self.pdf_file_name)

    def _write_outputs(self, pipe, image_dir, dump_flags, f_make_md_mode):
        f_dump_md, f_dump_middle_json, f_dump_model_json, f_dump_orig_pdf, f_dump_content_list = dump_flags
        
        if f_dump_md:
            md_content = pipe.get_markdown(image_dir, drop_mode=DropMode.NONE, make_mode=f_make_md_mode)
            self.md_writer.write_string(f'{self.pdf_file_name}.md', md_content)

        if f_dump_middle_json:
            self.md_writer.write_string(
                f'{self.pdf_file_name}_middle.json',
                json_parse.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4)
            )

        if f_dump_model_json:
            self.md_writer.write_string(
                f'{self.pdf_file_name}_model.json',
                json_parse.dumps(self.model_list, ensure_ascii=False, indent=4)
            )

        if f_dump_orig_pdf:
            self.md_writer.write(f'{self.pdf_file_name}_origin.pdf', self.pdf_bytes)

        if f_dump_content_list:
            content_list = pipe.get_content(image_dir, drop_mode=DropMode.NONE)
            self.md_writer.write_string(
                f'{self.pdf_file_name}_content_list.json',
                json_parse.dumps(content_list, ensure_ascii=False, indent=4)
            )

    def parse(self, 
             drawing_config: DrawingConfig = DrawingConfig(),
             output_config: OutputConfig = OutputConfig(),
             processing_config: ProcessingConfig = ProcessingConfig()):
        
        f_draw_model_bbox, f_draw_line_sort_bbox = self._setup_debug_mode(
            processing_config.debug_able,
            drawing_config.draw_model_bbox,
            drawing_config.draw_line_sort_bbox
        )

        self.prepare_env()
        self.convert_pdf_bytes(processing_config.start_page_id, processing_config.end_page_id)
        
        pipe = self._process_pdf(
            processing_config.lang,
            processing_config.layout_model,
            processing_config.formula_enable,
            processing_config.table_enable
        )
        image_dir = str(os.path.basename(self.local_image_dir))

        self._handle_drawings(
            pipe.pdf_mid_data['pdf_info'],
            self.model_list,
            (drawing_config.draw_layout_bbox,
             drawing_config.draw_span_bbox,
             f_draw_model_bbox,
             f_draw_line_sort_bbox)
        )

        self._write_outputs(
            pipe,
            image_dir,
            (output_config.dump_md,
             output_config.dump_middle_json,
             output_config.dump_model_json,
             output_config.dump_orig_pdf,
             output_config.dump_content_list),
            output_config.make_md_mode
        )

        logger.info(f'le répertoire de sortie local est {self.local_md_dir}')
