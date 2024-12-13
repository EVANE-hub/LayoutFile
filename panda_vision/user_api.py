"""Entrées utilisateur : tableau de modèles, chaque élément représente une page, chemin PDF dans S3, emplacement de sauvegarde des captures d'écran dans S3.

Ensuite :
    1) À partir du chemin S3, appeler l'API du cluster Spark pour obtenir ak, sk, endpoint et construire s3PDFReader
    2) À partir de l'adresse S3 fournie par l'utilisateur, appeler l'API du cluster Spark pour obtenir ak, sk, endpoint et construire s3ImageWriter

Le reste concernant la construction de s3cli et l'obtention de ak, sk est fait dans code-clean. Pas de dépendance inverse !!!
"""

from loguru import logger

from panda_vision.data.data_reader_writer import DataWriter
from panda_vision.utils.version import __version__
from panda_vision.model.doc_analyze_by_custom_model import doc_analyze
from panda_vision.pdf_parse_by_ocr import parse_pdf_by_ocr
from panda_vision.pdf_parse_by_txt import parse_pdf_by_txt

PARSE_TYPE_TXT = 'txt'
PARSE_TYPE_OCR = 'ocr'


def parse_txt_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                  start_page_id=0, end_page_id=None, lang=None,
                  *args, **kwargs):
    """Analyse des PDF textuels."""
    pdf_info_dict = parse_pdf_by_txt(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
        lang=lang,
    )

    pdf_info_dict['_parse_type'] = PARSE_TYPE_TXT

    pdf_info_dict['_version_name'] = __version__

    if lang is not None:
        pdf_info_dict['_lang'] = lang

    return pdf_info_dict


def parse_ocr_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                  start_page_id=0, end_page_id=None, lang=None,
                  *args, **kwargs):
    """Analyse des PDF par OCR."""
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
        lang=lang,
    )

    pdf_info_dict['_parse_type'] = PARSE_TYPE_OCR

    pdf_info_dict['_version_name'] = __version__

    if lang is not None:
        pdf_info_dict['_lang'] = lang

    return pdf_info_dict


def parse_union_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                    input_model_is_empty: bool = False,
                    start_page_id=0, end_page_id=None, lang=None,
                    *args, **kwargs):
    """Analyse complète des PDF mixtes (OCR et texte)."""

    def parse_pdf(method):
        try:
            return method(
                pdf_bytes,
                pdf_models,
                imageWriter,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                debug_mode=is_debug,
                lang=lang,
            )
        except Exception as e:
            logger.exception(e)
            return None

    pdf_info_dict = parse_pdf(parse_pdf_by_txt)
    if pdf_info_dict is None or pdf_info_dict.get('_need_drop', False):
        logger.warning('Échec ou erreur de parse_pdf_by_txt, passage à parse_pdf_by_ocr')
        if input_model_is_empty:
            layout_model = kwargs.get('layout_model', None)
            formula_enable = kwargs.get('formula_enable', None)
            table_enable = kwargs.get('table_enable', None)
            pdf_models = doc_analyze(
                pdf_bytes,
                ocr=True,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        pdf_info_dict = parse_pdf(parse_pdf_by_ocr)
        if pdf_info_dict is None:
            raise Exception('Les analyses parse_pdf_by_txt et parse_pdf_by_ocr ont échoué.')
        else:
            pdf_info_dict['_parse_type'] = PARSE_TYPE_OCR
    else:
        pdf_info_dict['_parse_type'] = PARSE_TYPE_TXT

    pdf_info_dict['_version_name'] = __version__

    if lang is not None:
        pdf_info_dict['_lang'] = lang

    return pdf_info_dict
