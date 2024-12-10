# flake8: noqa
import os
import time

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # désactiver la vérification des mises à jour albumentations
os.environ['YOLO_VERBOSE'] = 'False'  # désactiver le logger yolo

try:
    import torchtext

    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

from panda_vision.config.constants import *
from panda_vision.model.model_list import AtomicModel
from panda_vision.model.sub_modules.model_init import AtomModelSingleton
from panda_vision.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
from panda_vision.model.sub_modules.ocr.paddleocr.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)


class CustomPEKModel:

    def __init__(self, ocr: bool = False, show_log: bool = False, **kwargs):
        """
        ======== initialisation du modèle ========
        """
        # Obtenir le chemin absolu du fichier actuel (pdf_extract_kit.py)
        current_file_path = os.path.abspath(__file__)
        # Obtenir le répertoire du fichier actuel (model)
        current_dir = os.path.dirname(current_file_path)
        # Répertoire parent (magic_pdf)
        root_dir = os.path.dirname(current_dir)
        # Répertoire model_config
        model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
        # Construire le chemin complet du fichier model_configs.yaml
        config_path = os.path.join(model_config_dir, 'model_configs.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # Initialiser la configuration d'analyse

        # Configuration de la mise en page
        self.layout_config = kwargs.get('layout_config')
        self.layout_model_name = self.layout_config.get(
            'model', MODEL_NAME.DocLayout_YOLO
        )

        # Configuration des formules
        self.formula_config = kwargs.get('formula_config')
        self.mfd_model_name = self.formula_config.get(
            'mfd_model', MODEL_NAME.YOLO_V8_MFD
        )
        self.mfr_model_name = self.formula_config.get(
            'mfr_model', MODEL_NAME.UniMerNet_v2_Small
        )
        self.apply_formula = self.formula_config.get('enable', True)

        # Configuration des tableaux
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', False)
        self.table_max_time = self.table_config.get('max_time', TABLE_MAX_TIME_VALUE)
        self.table_model_name = self.table_config.get('model', MODEL_NAME.RAPID_TABLE)

        # Configuration OCR
        self.apply_ocr = ocr
        self.lang = kwargs.get('lang', None)

        logger.info(
            'Initialisation DocAnalysis, cela peut prendre du temps, modèle de mise en page: {}, formules: {}, ocr: {}, '
            'tableaux: {}, modèle de tableau: {}, langue: {}'.format(
                self.layout_model_name,
                self.apply_formula,
                self.apply_ocr,
                self.apply_table,
                self.table_model_name,
                self.lang,
            )
        )
        # Initialiser la solution d'analyse
        self.device = kwargs.get('device', 'cpu')
        logger.info('utilisation du périphérique: {}'.format(self.device))
        models_dir = kwargs.get(
            'models_dir', os.path.join(root_dir, 'resources', 'models')
        )
        logger.info('utilisation du répertoire de modèles: {}'.format(models_dir))

        atom_model_manager = AtomModelSingleton()

        # Initialisation de la reconnaissance des formules
        if self.apply_formula:
            # Initialiser le modèle de détection des formules
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.mfd_model_name]
                    )
                ),
                device=self.device,
            )

            # Initialiser le modèle d'analyse des formules
            mfr_weight_dir = str(
                os.path.join(models_dir, self.configs['weights'][self.mfr_model_name])
            )
            mfr_cfg_path = str(os.path.join(model_config_dir, 'UniMERNet', 'demo.yaml'))
            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=mfr_weight_dir,
                mfr_cfg_path=mfr_cfg_path,
                device=self.device,
            )

        # Initialisation du modèle de mise en page
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.LAYOUTLMv3,
                layout_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                layout_config_file=str(
                    os.path.join(
                        model_config_dir, 'layoutlmv3', 'layoutlmv3_base_inference.yaml'
                    )
                ),
                device=self.device,
            )
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=str(
                    os.path.join(
                        models_dir, self.configs['weights'][self.layout_model_name]
                    )
                ),
                device=self.device,
            )
        # Initialisation OCR
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            ocr_show_log=show_log,
            det_db_box_thresh=0.3,
            lang=self.lang
        )
        # Initialisation du modèle de tableau
        if self.apply_table:
            table_model_dir = self.configs['weights'][self.table_model_name]
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
                table_model_name=self.table_model_name,
                table_model_path=str(os.path.join(models_dir, table_model_dir)),
                table_max_time=self.table_max_time,
                device=self.device,
            )

        logger.info('Initialisation DocAnalysis terminée!')

    def __call__(self, image):

        # Détection de la mise en page
        layout_start = time.time()
        layout_res = []
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            layout_res = self.layout_model(image, ignore_catids=[])
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_res = self.layout_model.predict(image)
        layout_cost = round(time.time() - layout_start, 2)
        logger.info(f'temps de détection de mise en page: {layout_cost}')

        pil_img = Image.fromarray(image)

        if self.apply_formula:
            # Détection des formules
            mfd_start = time.time()
            mfd_res = self.mfd_model.predict(image)
            logger.info(f'temps mfd: {round(time.time() - mfd_start, 2)}')

            # Reconnaissance des formules
            mfr_start = time.time()
            formula_list = self.mfr_model.predict(mfd_res, image)
            layout_res.extend(formula_list)
            mfr_cost = round(time.time() - mfr_start, 2)
            logger.info(f'nombre de formules: {len(formula_list)}, temps mfr: {mfr_cost}')

        # Nettoyage de la mémoire vidéo
        clean_vram(self.device, vram_threshold=8)

        # Obtenir les zones OCR, tableaux et formules depuis layout_res
        ocr_res_list, table_res_list, single_page_mfdetrec_res = (
            get_res_list_from_layout_res(layout_res)
        )

        # Reconnaissance OCR
        ocr_start = time.time()
        # Traiter chaque zone nécessitant un traitement OCR
        for res in ocr_res_list:
            new_image, useful_list = crop_img(res, pil_img, crop_paste_x=50, crop_paste_y=50)
            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list)

            # Reconnaissance OCR
            new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
            if self.apply_ocr:
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res)[0]
            else:
                ocr_res = self.ocr_model.ocr(new_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]

            # Intégration des résultats
            if ocr_res:
                ocr_result_list = get_ocr_result_list(ocr_res, useful_list)
                layout_res.extend(ocr_result_list)

        ocr_cost = round(time.time() - ocr_start, 2)
        if self.apply_ocr:
            logger.info(f"temps ocr: {ocr_cost}")
        else:
            logger.info(f"temps de détection: {ocr_cost}")

        # Reconnaissance des tableaux
        if self.apply_table:
            table_start = time.time()
            for res in table_res_list:
                new_image, _ = crop_img(res, pil_img)
                single_table_start_time = time.time()
                html_code = None
                if self.table_model_name == MODEL_NAME.STRUCT_EQTABLE:
                    with torch.no_grad():
                        table_result = self.table_model.predict(new_image, 'html')
                        if len(table_result) > 0:
                            html_code = table_result[0]
                elif self.table_model_name == MODEL_NAME.TABLE_MASTER:
                    html_code = self.table_model.img2html(new_image)
                elif self.table_model_name == MODEL_NAME.RAPID_TABLE:
                    html_code, table_cell_bboxes, elapse = self.table_model.predict(
                        new_image
                    )
                run_time = time.time() - single_table_start_time
                if run_time > self.table_max_time:
                    logger.warning(
                        f'le traitement de reconnaissance de tableau dépasse le temps maximum {self.table_max_time}s'
                    )
                # Vérifier si le retour est normal
                if html_code:
                    expected_ending = html_code.strip().endswith(
                        '</html>'
                    ) or html_code.strip().endswith('</table>')
                    if expected_ending:
                        res['html'] = html_code
                    else:
                        logger.warning(
                            'échec du traitement de reconnaissance de tableau, fin HTML attendue non trouvée'
                        )
                else:
                    logger.warning(
                        'échec du traitement de reconnaissance de tableau, pas de retour html'
                    )
            logger.info(f'temps de tableau: {round(time.time() - table_start, 2)}')

        return layout_res
