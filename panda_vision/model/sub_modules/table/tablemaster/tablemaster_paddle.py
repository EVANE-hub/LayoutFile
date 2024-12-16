import os
from typing import Union, Dict

import cv2
import numpy as np
from PIL import Image
from ppstructure.table.predict_table import TableSystem
from ppstructure.utility import init_args

from panda_vision.config.constants import *

class TableMasterPaddleModel:
    """Module de conversion d'images de tableaux en format HTML.
    
    Ce modèle utilise PaddleOCR pour:
    1. Détecter la structure du tableau
    2. Reconnaître le texte dans chaque cellule
    3. Générer une représentation HTML du tableau
    """
    
    def __init__(self, config: Dict):
        """Initialise le modèle avec la configuration fournie.
        
        Args:
            config (Dict): Dictionnaire contenant:
                - model_dir: Chemin vers les modèles pré-entraînés
                - device: Dispositif d'exécution ('cpu' ou 'cuda')
                - table_max_len: Longueur maximale du tableau (optionnel)
        """
        self.table_sys = TableSystem(self.parse_args(**config))

    def img2html(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Convertit une image de tableau en HTML.
        
        Args:
            image: Image source au format PIL.Image ou numpy.ndarray
            
        Returns:
            str: Structure HTML du tableau avec son contenu
            
        Note:
            Si l'image est au format PIL, elle est convertie en BGR pour
            être compatible avec le modèle PaddleOCR.
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return self.table_sys(image)[0]['html']

    def parse_args(self, **kwargs) -> object:
        """Configure les paramètres du modèle.
        
        Args:
            **kwargs: Arguments de configuration incluant:
                - model_dir: Répertoire des modèles
                - device: Type de processeur (cpu/cuda)
                - table_max_len: Longueur max du tableau
                
        Returns:
            object: Arguments parsés pour TableSystem
            
        Note:
            Les chemins des modèles sont construits relativement au model_dir:
            - TableMaster pour la structure
            - Detectron pour la détection
            - Recognition pour la reconnaissance de texte
        """
        parser = init_args()
        model_dir = kwargs.get('model_dir')
        device = kwargs.get('device', 'cpu')
        
        config = {
            'use_gpu': device.startswith('cuda'),
            'table_max_len': kwargs.get('table_max_len', TABLE_MAX_LEN),
            'table_algorithm': 'TableMaster',
            'table_model_dir': os.path.join(model_dir, TABLE_MASTER_DIR),
            'table_char_dict_path': os.path.join(model_dir, TABLE_MASTER_DICT),
            'det_model_dir': os.path.join(model_dir, DETECT_MODEL_DIR),
            'rec_model_dir': os.path.join(model_dir, REC_MODEL_DIR),
            'rec_char_dict_path': os.path.join(model_dir, REC_CHAR_DICT),
        }
        
        parser.set_defaults(**config)
        return parser.parse_args([])
