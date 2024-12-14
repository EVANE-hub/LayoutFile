from enum import Enum
from typing import Optional
from loguru import logger
import json

from panda_vision.config.make_content_config import DropMode, MakeMode 
from panda_vision.data.data_reader_writer import DataWriter
from panda_vision.model.doc_analyze_by_custom_model import doc_analyze
from panda_vision.utils.json_compressor import JsonCompressor
from panda_vision.config.drop_reason import DropReason
from panda_vision.filter.pdf_meta_scan import pdf_meta_scan
from panda_vision.filter.pdf_classify_by_type import classify
from panda_vision.dict2md.ocr_mkcontent import union_make
from panda_vision.preprocessor.pdf_parse_by_txt import parse_pdf_by_txt
from panda_vision.preprocessor.pdf_parse_by_ocr import parse_pdf_by_ocr


from panda_vision.utils.version import __version__

class PDFType(Enum):
    OCR = 'ocr'
    TEXT = 'txt'
    UNION = 'union'

class PDFProcessor:
    def __init__(self, 
                 pdf_bytes: bytes,
                 model_list: list,
                 image_writer: DataWriter,
                 pdf_type: Optional[PDFType] = None,
                 is_debug: bool = False,
                 start_page_id: int = 0,
                 end_page_id: Optional[int] = None,
                 lang: Optional[str] = None,
                 layout_model: Optional[str] = None,
                 formula_enable: Optional[bool] = None, 
                 table_enable: Optional[bool] = None):

        self.pdf_bytes = pdf_bytes
        self.model_list = model_list
        self.image_writer = image_writer
        self.pdf_type = pdf_type
        self.pdf_mid_data = None
        self.is_debug = is_debug
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.lang = lang
        self.layout_model = layout_model
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.input_model_is_empty = len(self.model_list) == 0
    @staticmethod
    def classify_pdf(pdf_bytes: bytes) -> PDFType:
        """Détermine le type de PDF basé sur ses métadonnées"""
        pdf_meta = pdf_meta_scan(pdf_bytes)
        
        if pdf_meta.get('_need_drop', False):
            raise ValueError(f"PDF meta_scan need_drop, reason: {pdf_meta['_drop_reason']}")
            
        if pdf_meta['is_encrypted'] or pdf_meta['is_needs_password']:
            raise ValueError(f'PDF meta_scan need_drop, reason: {DropReason.ENCRYPTED}')
            
        is_text_pdf, _ = classify(
            pdf_meta['total_page'],
            pdf_meta['page_width_pts'],
            pdf_meta['page_height_pts'], 
            pdf_meta['image_info_per_page'],
            pdf_meta['text_len_per_page'],
            pdf_meta['imgs_per_page'],
            pdf_meta['text_layout_per_page'],
            pdf_meta['invalid_chars']
        )
        
        return PDFType.TEXT if is_text_pdf else PDFType.OCR

    def process(self) -> dict:
        """Traitement principal du PDF"""
        if not self.pdf_type:
            self.pdf_type = self.classify_pdf(self.pdf_bytes)

        is_ocr = self.pdf_type in (PDFType.OCR, PDFType.UNION)
        self.model_list = doc_analyze(
            self.pdf_bytes, 
            ocr=is_ocr,
            start_page_id=self.start_page_id,
            end_page_id=self.end_page_id,
            lang=self.lang,
            layout_model=self.layout_model,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable
        )

        try:
            if self.pdf_type == PDFType.TEXT:
                self.pdf_mid_data = self._parse_txt()
            elif self.pdf_type == PDFType.OCR:
                self.pdf_mid_data = self._parse_ocr()
            else: 
                self.pdf_mid_data = self._parse_union()
        except Exception as e:
            logger.exception(f"Erreur lors de l'analyse: {e}")
            raise

        return self.pdf_mid_data

    def _parse_txt(self) -> dict:
        """Analyse des PDF textuels"""
        
        result = parse_pdf_by_txt(
            self.pdf_bytes,
            self.model_list,
            self.image_writer,
            start_page_id=self.start_page_id,
            end_page_id=self.end_page_id,
            debug_mode=self.is_debug,
            lang=self.lang
        )
        
        self._add_metadata(result, 'txt')
        return result

    def _parse_ocr(self) -> dict:
        """Analyse des PDF par OCR"""
        
        result = parse_pdf_by_ocr(
            self.pdf_bytes,
            self.model_list, 
            self.image_writer,
            start_page_id=self.start_page_id,
            end_page_id=self.end_page_id,
            debug_mode=self.is_debug,
            lang=self.lang
        )
        
        self._add_metadata(result, 'ocr')
        return result

    def _parse_union(self) -> dict:
        """Analyse des PDF mixtes"""
        result = self._parse_txt()
        
        if result is None or result.get('_need_drop', False):
            logger.warning('Échec parse_txt, passage à parse_ocr')
            if self.input_model_is_empty:
                self.model_list = doc_analyze(
                    self.pdf_bytes,
                    ocr=True,
                    start_page_id=self.start_page_id,
                    end_page_id=self.end_page_id,
                    lang=self.lang,
                    layout_model=self.layout_model,
                    formula_enable=self.formula_enable,
                    table_enable=self.table_enable
                )
            result = self._parse_ocr()
            
        return result

    def _add_metadata(self, result: dict, parse_type: str):
        """Ajoute les métadonnées au résultat"""
        result['_parse_type'] = parse_type
        result['_version_name'] = __version__
        if self.lang:
            result['_lang'] = self.lang

    def get_content(self, img_parent_path: str, drop_mode: DropMode = DropMode.WHOLE_PDF) -> list:
        """Génère le contenu au format unifié"""
        compressed = JsonCompressor.compress_json(self.pdf_mid_data)
        pdf_data = JsonCompressor.decompress_json(compressed)
        return union_make(pdf_data['pdf_info'], MakeMode.STANDARD_FORMAT, drop_mode, img_parent_path)

    def get_markdown(self, img_parent_path: str, 
                    drop_mode: DropMode = DropMode.WHOLE_PDF,
                    make_mode: MakeMode = MakeMode.MM_MD) -> list:
        """Génère le contenu au format Markdown"""
        compressed = JsonCompressor.compress_json(self.pdf_mid_data)
        pdf_data = JsonCompressor.decompress_json(compressed)
        return union_make(pdf_data['pdf_info'], make_mode, drop_mode, img_parent_path)