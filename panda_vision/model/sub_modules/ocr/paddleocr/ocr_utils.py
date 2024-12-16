import cv2
import numpy as np
from loguru import logger
from io import BytesIO
from PIL import Image
import base64
from panda_vision.utils.boxbase import __is_overlaps_y_exceeds_threshold
from panda_vision.preprocessor.ocr_dict_merge import merge_spans_to_line

from ppocr.utils.utility import check_and_read


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes),
                                      encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def bbox_to_points(bbox):
    """ Convertir le format bbox en un tableau de quatre sommets """
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype('float32')


def points_to_bbox(points):
    """ Convertir un tableau de quatre sommets en format bbox """
    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return [x0, y0, x1, y1]


def merge_intervals(intervals):
    # Trier les intervalles en fonction de la valeur de début
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # Si la liste des intervalles fusionnés est vide ou si l'intervalle actuel
        # ne chevauche pas le précédent, il suffit de l'ajouter.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Sinon, il y a chevauchement, donc nous fusionnons les intervalles actuel et précédent.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def remove_intervals(original, masks):
    # Fusionner tous les intervalles de masque
    merged_masks = merge_intervals(masks)

    result = []
    original_start, original_end = original

    for mask in merged_masks:
        mask_start, mask_end = mask

        # Si le masque commence après la plage originale, l'ignorer
        if mask_start > original_end:
            continue

        # Si le masque se termine avant le début de la plage originale, l'ignorer
        if mask_end < original_start:
            continue

        # Supprimer la partie masquée de la plage originale
        if original_start < mask_start:
            result.append([original_start, mask_start - 1])

        original_start = max(mask_end + 1, original_start)

    # Ajouter la partie restante de la plage originale, le cas échéant
    if original_start <= original_end:
        result.append([original_start, original_end])

    return result


def update_det_boxes(dt_boxes, mfd_res):
    new_dt_boxes = []
    angle_boxes_list = []
    for text_box in dt_boxes:

        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box)
            continue

        text_bbox = points_to_bbox(text_box)
        masks_list = []
        for mf_box in mfd_res:
            mf_bbox = mf_box['bbox']
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                masks_list.append([mf_bbox[0], mf_bbox[2]])
        text_x_range = [text_bbox[0], text_bbox[2]]
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        temp_dt_box = []
        for text_remove_mask in text_remove_mask_range:
            temp_dt_box.append(bbox_to_points([text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3]]))
        if len(temp_dt_box) > 0:
            new_dt_boxes.extend(temp_dt_box)

    new_dt_boxes.extend(angle_boxes_list)

    return new_dt_boxes


def merge_overlapping_spans(spans):
    """
    Fusionne les intervalles qui se chevauchent sur la même ligne.

    :param spans: Une liste de coordonnées d'intervalles [(x1, y1, x2, y2), ...]
    :return: Une liste d'intervalles fusionnés
    """
    # Retourner une liste vide si la liste d'intervalles d'entrée est vide
    if not spans:
        return []

    # Trier les intervalles par leur coordonnée x de départ
    spans.sort(key=lambda x: x[0])

    # Initialiser la liste des intervalles fusionnés
    merged = []
    for span in spans:
        # Décompresser les coordonnées de l'intervalle
        x1, y1, x2, y2 = span
        # Si la liste fusionnée est vide ou s'il n'y a pas de chevauchement horizontal, ajouter l'intervalle directement
        if not merged or merged[-1][2] < x1:
            merged.append(span)
        else:
            # S'il y a chevauchement horizontal, fusionner l'intervalle actuel avec le précédent
            last_span = merged.pop()
            # Mettre à jour le coin supérieur gauche de l'intervalle fusionné avec le plus petit (x1, y1) et le coin inférieur droit avec le plus grand (x2, y2)
            x1 = min(last_span[0], x1)
            y1 = min(last_span[1], y1)
            x2 = max(last_span[2], x2)
            y2 = max(last_span[3], y2)
            # Ajouter l'intervalle fusionné à la liste
            merged.append((x1, y1, x2, y2))

    # Retourner la liste des intervalles fusionnés
    return merged


def merge_det_boxes(dt_boxes):
    """
    Fusionner les boîtes de détection.

    Cette fonction prend une liste de boîtes de détection, chacune représentée par quatre points d'angle.
    L'objectif est de fusionner ces boîtes en régions de texte plus grandes.

    Paramètres:
    dt_boxes (list): Une liste contenant plusieurs boîtes de détection de texte, où chaque boîte est définie par quatre points d'angle.

    Retourne:
    list: Une liste contenant les régions de texte fusionnées, où chaque région est représentée par quatre points d'angle.
    """
    # Convertir les boîtes de détection en un format de dictionnaire avec des boîtes et des types
    dt_boxes_dict_list = []
    angle_boxes_list = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)

        if calculate_is_angle(text_box):
            angle_boxes_list.append(text_box)
            continue

        text_box_dict = {
            'bbox': text_bbox,
            'type': 'text',
        }
        dt_boxes_dict_list.append(text_box_dict)

    # Fusionner les régions de texte adjacentes en lignes
    lines = merge_spans_to_line(dt_boxes_dict_list)

    # Initialiser une nouvelle liste pour stocker les régions de texte fusionnées
    new_dt_boxes = []
    for line in lines:
        line_bbox_list = []
        for span in line:
            line_bbox_list.append(span['bbox'])

        # Fusionner les régions de texte qui se chevauchent dans la même ligne
        merged_spans = merge_overlapping_spans(line_bbox_list)

        # Convertir les régions de texte fusionnées en format de points et les ajouter à la nouvelle liste de boîtes de détection
        for span in merged_spans:
            new_dt_boxes.append(bbox_to_points(span))

    new_dt_boxes.extend(angle_boxes_list)

    return new_dt_boxes


def get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list):
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    # Ajuster les coordonnées de la zone de formule
    adjusted_mfdetrec_res = []
    for mf_res in single_page_mfdetrec_res:
        mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
        # Ajuster les coordonnées de la zone de formule aux coordonnées relatives à la zone de recadrage
        x0 = mf_xmin - xmin + paste_x
        y0 = mf_ymin - ymin + paste_y
        x1 = mf_xmax - xmin + paste_x
        y1 = mf_ymax - ymin + paste_y
        # Filtrer les blocs de formule en dehors du graphique
        if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
            continue
        else:
            adjusted_mfdetrec_res.append({
                "bbox": [x0, y0, x1, y1],
            })
    return adjusted_mfdetrec_res


def get_ocr_result_list(ocr_res, useful_list):
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    ocr_result_list = []
    for box_ocr_res in ocr_res:

        if len(box_ocr_res) == 2:
            p1, p2, p3, p4 = box_ocr_res[0]
            text, score = box_ocr_res[1]
            # logger.info(f"text: {text}, score: {score}")
            if score < 0.6:  # Filtrer les résultats de faible confiance
                continue
        else:
            p1, p2, p3, p4 = box_ocr_res
            text, score = "", 1
        # average_angle_degrees = calculate_angle_degrees(box_ocr_res[0])
        # if average_angle_degrees > 0.5:
        poly = [p1, p2, p3, p4]
        if calculate_is_angle(poly):
            # logger.info(f"average_angle_degrees: {average_angle_degrees}, text: {text}")
            # Si l'angle avec l'axe des x dépasse 0,5 degré, ajuster les bordures
            # Calculer le centre géométrique
            x_center = sum(point[0] for point in poly) / 4
            y_center = sum(point[1] for point in poly) / 4
            new_height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
            new_width = p3[0] - p1[0]
            p1 = [x_center - new_width / 2, y_center - new_height / 2]
            p2 = [x_center + new_width / 2, y_center - new_height / 2]
            p3 = [x_center + new_width / 2, y_center + new_height / 2]
            p4 = [x_center - new_width / 2, y_center + new_height / 2]

        # Convertir les coordonnées dans le système de coordonnées d'origine
        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]

        ocr_result_list.append({
            'category_id': 15,
            'poly': p1 + p2 + p3 + p4,
            'score': float(round(score, 2)),
            'text': text,
        })

    return ocr_result_list


def calculate_is_angle(poly):
    p1, p2, p3, p4 = poly
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    if 0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height:
        return False
    else:
        # logger.info((p3[1] - p1[1])/height)
        return True