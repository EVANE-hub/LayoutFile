import time

import fitz
import numpy as np
from loguru import logger

from panda_vision.libs.clean_memory import clean_memory
from panda_vision.libs.config_reader import get_local_models_dir, get_device, get_table_recog_config, get_layout_config, \
    get_formula_config
from panda_vision.model.model_list import MODEL
import panda_vision.model as model_config


def dict_compare(d1, d2):
    return d1.items() == d2.items()


def remove_duplicates_dicts(lst):
    unique_dicts = []
    for dict_item in lst:
        if not any(
                dict_compare(dict_item, existing_dict) for existing_dict in unique_dicts
        ):
            unique_dicts.append(dict_item)
    return unique_dicts


def load_images_from_pdf(pdf_bytes: bytes, dpi=200, start_page_id=0, end_page_id=None) -> list:
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow n'est pas installé, veuillez l'installer avec pip.")
        exit(1)

    images = []
    with fitz.open("pdf", pdf_bytes) as doc:
        pdf_page_num = doc.page_count
        end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
        if end_page_id > pdf_page_num - 1:
            logger.warning("end_page_id est hors limites, utilisation de la longueur des images")
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # Si la largeur ou la hauteur dépasse 4500 après mise à l'échelle, ne pas redimensionner davantage
                if pm.width > 4500 or pm.height > 4500:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
                img = np.array(img)
                img_dict = {"img": img, "width": pm.width, "height": pm.height}
            else:
                img_dict = {"img": [], "width": 0, "height": 0}

            images.append(img_dict)
    return images


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, ocr: bool, show_log: bool, lang=None, layout_model=None, formula_enable=None, table_enable=None):
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        if key not in self._models:
            self._models[key] = custom_model_init(ocr=ocr, show_log=show_log, lang=lang, layout_model=layout_model,
                                                  formula_enable=formula_enable, table_enable=table_enable)
        return self._models[key]


def custom_model_init(ocr: bool = False, show_log: bool = False, lang=None,
                      layout_model=None, formula_enable=None, table_enable=None):

    model = None

    if model_config.__model_mode__ == "lite":
        logger.warning("Le mode Lite est fourni uniquement pour les tests des développeurs, la qualité des résultats "
                       "n'est pas garantie.")
        model = MODEL.Paddle
    elif model_config.__model_mode__ == "full":
        model = MODEL.PEK

    if model_config.__use_inside_model__:
        model_init_start = time.time()
        if model == MODEL.Paddle:
            from panda_vision.model.pp_structure_v2 import CustomPaddleModel
            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:
            from panda_vision.model.pdf_extract_kit import CustomPEKModel
            # Lecture du répertoire des modèles et du périphérique depuis le fichier de configuration
            local_models_dir = get_local_models_dir()
            device = get_device()

            layout_config = get_layout_config()
            if layout_model is not None:
                layout_config["model"] = layout_model

            formula_config = get_formula_config()
            if formula_enable is not None:
                formula_config["enable"] = formula_enable

            table_config = get_table_recog_config()
            if table_enable is not None:
                table_config["enable"] = table_enable

            model_input = {
                            "ocr": ocr,
                            "show_log": show_log,
                            "models_dir": local_models_dir,
                            "device": device,
                            "table_config": table_config,
                            "layout_config": layout_config,
                            "formula_config": formula_config,
                            "lang": lang,
            }

            custom_model = CustomPEKModel(**model_input)
        else:
            logger.error("Nom de modèle non autorisé!")
            exit(1)
        model_init_cost = time.time() - model_init_start
        logger.info(f"temps d'initialisation du modèle: {model_init_cost}")
    else:
        logger.error("use_inside_model est False, impossible d'utiliser le modèle interne")
        exit(1)

    return custom_model


def doc_analyze(pdf_bytes: bytes, ocr: bool = False, show_log: bool = False,
                start_page_id=0, end_page_id=None, lang=None,
                layout_model=None, formula_enable=None, table_enable=None):

    if lang == "":
        lang = None

    model_manager = ModelSingleton()
    custom_model = model_manager.get_model(ocr, show_log, lang, layout_model, formula_enable, table_enable)

    with fitz.open("pdf", pdf_bytes) as doc:
        pdf_page_num = doc.page_count
        end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
        if end_page_id > pdf_page_num - 1:
            logger.warning("end_page_id est hors limites, utilisation de la longueur des images")
            end_page_id = pdf_page_num - 1

    images = load_images_from_pdf(pdf_bytes, start_page_id=start_page_id, end_page_id=end_page_id)

    model_json = []
    doc_analyze_start = time.time()

    for index, img_dict in enumerate(images):
        img = img_dict["img"]
        page_width = img_dict["width"]
        page_height = img_dict["height"]
        if start_page_id <= index <= end_page_id:
            page_start = time.time()
            result = custom_model(img)
            logger.info(f'-----page_id : {index}, temps total de la page: {round(time.time() - page_start, 2)}-----')
        else:
            result = []
        page_info = {"page_no": index, "height": page_height, "width": page_width}
        page_dict = {"layout_dets": result, "page_info": page_info}
        model_json.append(page_dict)

    gc_start = time.time()
    clean_memory()
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f"temps gc: {gc_time}")

    doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    doc_analyze_speed = round( (end_page_id + 1 - start_page_id) / doc_analyze_time, 2)
    logger.info(f"temps d'analyse du document: {round(time.time() - doc_analyze_start, 2)},"
                f" vitesse: {doc_analyze_speed} pages/seconde")

    return model_json
