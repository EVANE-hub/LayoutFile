import random

from loguru import logger

try:
    from paddleocr import PPStructure
except ImportError:
    logger.error('paddleocr non installé, veuillez installer avec "pip install panda-vision[lite]"')
    exit(1)


def region_to_bbox(region):
    x0 = region[0][0]
    y0 = region[0][1]
    x1 = region[2][0]
    y1 = region[2][1]
    return [x0, y0, x1, y1]


class CustomPaddleModel:
    def __init__(self,
                 ocr: bool = False,
                 show_log: bool = False,
                 lang=None,
                 det_db_box_thresh=0.3,
                 use_dilation=True,
                 det_db_unclip_ratio=1.8
    ):
        if lang is not None:
            self.model = PPStructure(table=False,
                                     ocr=True,
                                     show_log=show_log,
                                     lang=lang,
                                     det_db_box_thresh=det_db_box_thresh,
                                     use_dilation=use_dilation,
                                     det_db_unclip_ratio=det_db_unclip_ratio,
            )
        else:
            self.model = PPStructure(table=False,
                                     ocr=True,
                                     show_log=show_log,
                                     det_db_box_thresh=det_db_box_thresh,
                                     use_dilation=use_dilation,
                                     det_db_unclip_ratio=det_db_unclip_ratio,
            )

    def __call__(self, img):
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python non installé, veuillez installer avec pip.")
            exit(1)
        # Convertir l'image RGB en format BGR pour paddle
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self.model(img)
        spans = []
        for line in result:
            line.pop("img")
            """
            Adapter le type no. pour la sortie paddle
            title: 0 # titre
            text: 1 # texte
            header: 2 # abandonner
            footer: 2 # abandonner  
            reference: 1 # texte ou abandonner
            equation: 8 # équation entre les lignes block
            equation: 14 # équation entre les lignes text
            figure: 3 # image
            figure_caption: 4 # légende d'image
            table: 5 # tableau
            table_caption: 6 # légende de tableau
            """
            if line["type"] == "title":
                line["category_id"] = 0
            elif line["type"] in ["text", "reference"]:
                line["category_id"] = 1
            elif line["type"] == "figure":
                line["category_id"] = 3
            elif line["type"] == "figure_caption":
                line["category_id"] = 4
            elif line["type"] == "table":
                line["category_id"] = 5
            elif line["type"] == "table_caption":
                line["category_id"] = 6
            elif line["type"] == "equation":
                line["category_id"] = 8
            elif line["type"] in ["header", "footer"]:
                line["category_id"] = 2
            else:
                logger.warning(f"type inconnu: {line['type']}")

            # Compatible avec les versions de paddleocr qui ne produisent pas de score
            if line.get("score") is None:
                line["score"] = 0.5 + random.random() * 0.5

            res = line.pop("res", None)
            if res is not None and len(res) > 0:
                for span in res:
                    new_span = {
                        "category_id": 15,
                        "bbox": region_to_bbox(span["text_region"]),
                        "score": span["confidence"],
                        "text": span["text"],
                    }
                    spans.append(new_span)

        if len(spans) > 0:
            result.extend(spans)

        return result
