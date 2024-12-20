from abc import ABC, abstractmethod

from panda_vision.config.drop_reason import DropReason
from panda_vision.config.make_content_config import DropMode, MakeMode
from panda_vision.data.data_reader_writer import DataWriter
from panda_vision.dict2md.ocr_mkcontent import union_make
from panda_vision.filter.pdf_classify_by_type import classify
from panda_vision.filter.pdf_meta_scan import pdf_meta_scan
from panda_vision.libs.json_compressor import JsonCompressor


class AbsPipe(ABC):
    """Classe abstraite pour le traitement txt et ocr."""
    PIP_OCR = 'ocr'
    PIP_TXT = 'txt'

    def __init__(self, pdf_bytes: bytes, model_list: list, image_writer: DataWriter, is_debug: bool = False,
                 start_page_id=0, end_page_id=None, lang=None, layout_model=None, formula_enable=None, table_enable=None):
        self.pdf_bytes = pdf_bytes
        self.model_list = model_list
        self.image_writer = image_writer
        self.pdf_mid_data = None  # Non compressé
        self.is_debug = is_debug
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.lang = lang
        self.layout_model = layout_model
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    def get_compress_pdf_mid_data(self):
        return JsonCompressor.compress_json(self.pdf_mid_data)

    @abstractmethod
    def pipe_classify(self):
        """Classification avec état."""
        raise NotImplementedError

    @abstractmethod
    def pipe_analyze(self):
        """Analyse du modèle avec état."""
        raise NotImplementedError

    @abstractmethod
    def pipe_parse(self):
        """Analyse avec état."""
        raise NotImplementedError

    def pipe_mk_uni_format(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF):
        content_list = AbsPipe.mk_uni_format(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode)
        return content_list

    def pipe_mk_markdown(self, img_parent_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD):
        md_content = AbsPipe.mk_markdown(self.get_compress_pdf_mid_data(), img_parent_path, drop_mode, md_make_mode)
        return md_content

    @staticmethod
    def classify(pdf_bytes: bytes) -> str:
        """Détermine si le PDF est un PDF texte ou OCR en fonction des métadonnées."""
        pdf_meta = pdf_meta_scan(pdf_bytes)
        if pdf_meta.get('_need_drop', False):  # Si le drapeau de rejet est présent, lever une exception
            raise Exception(f"pdf meta_scan need_drop,reason is {pdf_meta['_drop_reason']}")
        else:
            is_encrypted = pdf_meta['is_encrypted']
            is_needs_password = pdf_meta['is_needs_password']
            if is_encrypted or is_needs_password:  # Ne pas traiter les PDFs cryptés, protégés par mot de passe ou sans pages
                raise Exception(f'pdf meta_scan need_drop,reason is {DropReason.ENCRYPTED}')
            else:
                is_text_pdf, results = classify(
                    pdf_meta['total_page'],
                    pdf_meta['page_width_pts'],
                    pdf_meta['page_height_pts'],
                    pdf_meta['image_info_per_page'],
                    pdf_meta['text_len_per_page'],
                    pdf_meta['imgs_per_page'],
                    pdf_meta['text_layout_per_page'],
                    pdf_meta['invalid_chars'],
                )
                if is_text_pdf:
                    return AbsPipe.PIP_TXT
                else:
                    return AbsPipe.PIP_OCR

    @staticmethod
    def mk_uni_format(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF) -> list:
        """Génère une liste de contenu au format unifié selon le type de PDF."""
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        pdf_info_list = pdf_mid_data['pdf_info']
        content_list = union_make(pdf_info_list, MakeMode.STANDARD_FORMAT, drop_mode, img_buket_path)
        return content_list

    @staticmethod
    def mk_markdown(compressed_pdf_mid_data: str, img_buket_path: str, drop_mode=DropMode.WHOLE_PDF, md_make_mode=MakeMode.MM_MD) -> list:
        """Génère du markdown selon le type de PDF."""
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        pdf_info_list = pdf_mid_data['pdf_info']
        md_content = union_make(pdf_info_list, md_make_mode, drop_mode, img_buket_path)
        return md_content
