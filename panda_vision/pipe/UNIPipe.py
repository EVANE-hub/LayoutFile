import json
from loguru import logger

from panda_vision.config.make_content_config import DropMode, MakeMode
from panda_vision.data.data_reader_writer import DataWriter
from panda_vision.model.doc_analyze_by_custom_model import doc_analyze
from panda_vision.pipe.AbsPipe import AbsPipe
from panda_vision.user_api import parse_ocr_pdf, parse_union_pdf


class UNIPipe(AbsPipe):

    def __init__(self, pdf_bytes: bytes, jso_useful_key: dict, image_writer: DataWriter, is_debug: bool = False,
                 start_page_id=0, end_page_id=None, lang=None,
                 layout_model=None, formula_enable=None, table_enable=None):
        self.pdf_type = jso_useful_key['_pdf_type']
        super().__init__(pdf_bytes, jso_useful_key['model_list'], image_writer, is_debug, start_page_id, end_page_id,
                         lang, layout_model, formula_enable, table_enable)
        self.input_model_is_empty = len(self.model_list) == 0

    def pipe_classify(self):
        self.pdf_type = AbsPipe.classify(self.pdf_bytes)

    def pipe_analyze(self):
        ocr = self.pdf_type == self.PIP_OCR
        self.model_list = doc_analyze(
            self.pdf_bytes, ocr=ocr,
            start_page_id=self.start_page_id, end_page_id=self.end_page_id,
            lang=self.lang, layout_model=self.layout_model,
            formula_enable=self.formula_enable, table_enable=self.table_enable
        )

    def pipe_parse(self):
        if self.pdf_type == self.PIP_TXT:
            self.pdf_mid_data = parse_union_pdf(
                self.pdf_bytes, self.model_list, self.image_writer,
                is_debug=self.is_debug, input_model_is_empty=self.input_model_is_empty,
                start_page_id=self.start_page_id, end_page_id=self.end_page_id,
                lang=self.lang, layout_model=self.layout_model,
                formula_enable=self.formula_enable, table_enable=self.table_enable
            )
        elif self.pdf_type == self.PIP_OCR:
            self.pdf_mid_data = parse_ocr_pdf(
                self.pdf_bytes, self.model_list, self.image_writer,
                is_debug=self.is_debug,
                start_page_id=self.start_page_id, end_page_id=self.end_page_id,
                lang=self.lang
            )

    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.NONE_WITH_REASON):
        result = super().pipe_mk_uni_format(img_parent_path, drop_mode)
        logger.info('uni_pipe mk content list finished')
        return result

    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        result = super().pipe_mk_markdown(img_parent_path, drop_mode, md_make_mode)
        logger.info(f'uni_pipe mk {md_make_mode} finished')
        return result
