import cv2
import random
from loguru import logger
from typing import List, Dict, Any

try:
    from paddleocr import PPStructure
except ImportError:
    logger.error("paddleocr non installé, veuillez installer avec pip.")
    exit(1)

# Constants pour les catégories
CATEGORY_MAPPING = {
    "title": 0,
    "text": 1,
    "reference": 1,
    "header": 2,
    "footer": 2,
    "figure": 3,
    "figure_caption": 4,
    "table": 5,
    "table_caption": 6,
    "equation": 8
}

def region_to_bbox(region: List[List[int]]) -> List[int]:
    """Convertit une région en bbox [x0, y0, x1, y1]."""
    return [region[0][0], region[0][1], region[2][0], region[2][1]]

class CustomPaddleModel:
    """Modèle personnalisé utilisant PPStructure pour la détection de layout."""
    
    def __init__(self,
                 ocr: bool = False,
                 show_log: bool = False,
                 lang: str = None,
                 det_db_box_thresh: float = 0.3,
                 use_dilation: bool = True,
                 det_db_unclip_ratio: float = 1.8
    ):
        """Initialise le modèle PPStructure avec les paramètres spécifiés."""
        model_params = {
            "table": False,
            "ocr": True,
            "show_log": show_log,
            "det_db_box_thresh": det_db_box_thresh,
            "use_dilation": use_dilation,
            "det_db_unclip_ratio": det_db_unclip_ratio
        }
        
        if lang is not None:
            model_params["lang"] = lang
            
        self.model = PPStructure(**model_params)

    def __call__(self, img) -> List[Dict[str, Any]]:
        """Traite l'image et retourne les résultats de détection."""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self.model(img)
        spans = []

        for line in result:
            line.pop("img")
            
            # Définir category_id basé sur le type
            line_type = line["type"]
            line["category_id"] = CATEGORY_MAPPING.get(line_type, None)
            
            if line["category_id"] is None:
                logger.warning(f"type inconnu: {line_type}")

            # Gérer le score si non présent
            if "score" not in line:
                line["score"] = 0.5 + random.random() * 0.5

            # Traiter les spans si présents
            if res := line.pop("res", None):
                spans.extend([{
                    "category_id": 15,
                    "bbox": region_to_bbox(span["text_region"]),
                    "score": span["confidence"],
                    "text": span["text"]
                } for span in res])

        if spans:
            result.extend(spans)

        return result
