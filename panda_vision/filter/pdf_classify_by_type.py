"""
Classifie les PDF comme texte ou scan en fonction des résultats de meta_scan.
Critères de classification:
I. Un PDF est considéré comme texte s'il remplit l'une des conditions suivantes:
  1. Sur N pages aléatoires, au moins une page contient plus de 100 caractères
  2. Au moins une page ne contient aucune image
II. Un PDF est considéré comme scan s'il remplit l'une des conditions suivantes:
  1. La plupart des pages ont des textes de longueur identique.
"""

import json
import sys
from collections import Counter

import click
import numpy as np
from loguru import logger

from panda_vision.utils.commons import mymax, get_top_percent_list
from panda_vision.filter.pdf_meta_scan import scan_max_page, junk_limit_min

TEXT_LEN_THRESHOLD = 100
AVG_TEXT_LEN_THRESHOLD = 100
TEXT_LEN_SAMPLE_RATIO = 0.1

def merge_images(image_list, page_width, page_height, max_offset=5, max_gap=2):
    """Fusionne les images qui se chevauchent ou sont proches sur une page.
    
    Args:
        image_list: Liste des images par page, chaque image contient [x0,y0,x1,y1,id]
        page_width: Largeur de la page
        page_height: Hauteur de la page
        max_offset: Décalage maximum autorisé pour la fusion
        max_gap: Écart maximum autorisé entre les images
        
    Returns:
        Liste des images fusionnées par page
    """
    image_list_result = []
    for page_images in image_list:
        page_result = []
        dedup = set()
        for img in page_images:
            x0, y0, x1, y1, img_bojid = img
            if (x0, y0, x1, y1) in dedup:
                continue
            dedup.add((x0, y0, x1, y1))
            page_result.append([x0, y0, x1, y1, img_bojid])
        image_list_result.append(page_result)

    merged_images = []
    for page_images in image_list_result:
        if not page_images:
            continue

        page_images.sort(key=lambda img: (img[1], img[0]))
        merged = [page_images[0]]

        for img in page_images[1:]:
            x0, y0, x1, y1, imgid = img
            last_img = merged[-1]
            last_x0, last_y0, last_x1, last_y1, last_imgid = last_img

            full_width = abs(x1 - x0) >= page_width * 0.9
            full_height = abs(y1 - y0) >= page_height * 0.9

            close1 = (last_x0 - max_offset) <= x0 <= (last_x0 + max_offset) and (last_x1 - max_offset) <= x1 <= (
                        last_x1 + max_offset) and (last_y1 - max_gap) <= y0 <= (last_y1 + max_gap)

            close2 = (last_y0 - max_offset) <= y0 <= (last_y0 + max_offset) and (last_y1 - max_offset) <= y1 <= (
                        last_y1 + max_offset) and (last_x1 - max_gap) <= x0 <= (last_x1 + max_gap)

            if (full_width and close1) or (full_height and close2):
                merged[-1] = [min(x0, last_x0), min(y0, last_y0), max(x1, last_x1), max(y1, last_y1), imgid]
            else:
                merged.append(img)

        merged_images.append(merged)

    return merged_images

def classify_by_area(total_page: int, page_width, page_height, img_sz_list, text_len_list: list):
    """Classifie le PDF selon la surface des images.
    
    Args:
        total_page: Nombre total de pages
        page_width: Largeur de la page
        page_height: Hauteur de la page
        img_sz_list: Liste des tailles d'images par page
        text_len_list: Liste des longueurs de texte par page
        
    Returns:
        True si le PDF est considéré comme texte, False sinon
    """
    objid_cnt = Counter([objid for page_img_sz in img_sz_list for _, _, _, _, objid in page_img_sz])
    if total_page >= scan_max_page:
        total_page = scan_max_page

    repeat_threshold = 2
    bad_image_objid = set([objid for objid, cnt in objid_cnt.items() if cnt >= repeat_threshold])

    img_sz_list = [[img_sz for img_sz in page_img_sz if img_sz[-1] not in bad_image_objid] for page_img_sz in img_sz_list]
    img_sz_list = merge_images(img_sz_list, page_width, page_height)

    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in img_sz_list]
    page_area = page_width * page_height
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.5]

    return len(max_image_area_per_page) < 0.5 * total_page

def classify_by_text_len(text_len_list: list, total_page: int):
    """Classifie le PDF selon la longueur du texte sur un échantillon de pages.
    
    Args:
        text_len_list: Liste des longueurs de texte par page
        total_page: Nombre total de pages
        
    Returns:
        True si au moins une page contient suffisamment de texte
    """
    select_page_cnt = int(total_page * TEXT_LEN_SAMPLE_RATIO)
    if select_page_cnt < 5:
        select_page_cnt = total_page

    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    text_len_lst = [text_len_list[i] for i in page_num]
    return any([text_len > TEXT_LEN_THRESHOLD for text_len in text_len_lst])

def classify_by_avg_words(text_len_list: list):
    """Classifie le PDF selon la moyenne des mots par page.
    
    Args:
        text_len_list: Liste des longueurs de texte par page
        
    Returns:
        True si la moyenne des mots dépasse le seuil
    """
    sum_words = sum(text_len_list)
    count_of_numbers = len(text_len_list)
    if count_of_numbers == 0:
        return False
    avg_words = round(sum_words / count_of_numbers)
    return avg_words > AVG_TEXT_LEN_THRESHOLD

def classify_by_img_num(img_sz_list: list, img_num_list: list):
    """Classifie le PDF selon le nombre d'images.
    
    Args:
        img_sz_list: Liste des tailles d'images par page
        img_num_list: Liste du nombre d'images par page
        
    Returns:
        True si la distribution des images n'indique pas un scan
    """
    count_img_sz_list_not_none = sum(1 for item in img_sz_list if item)
    top_eighty_percent = get_top_percent_list(img_num_list, 0.8)
    return not (count_img_sz_list_not_none <= 1 and len(set(top_eighty_percent)) == 1 and max(img_num_list) >= junk_limit_min)

def classify_by_text_layout(text_layout_per_page: list):
    """Classifie le PDF selon l'orientation du texte.
    
    Args:
        text_layout_per_page: Liste des orientations de texte par page
        
    Returns:
        True si le texte est principalement horizontal
    """
    count_vertical = sum(1 for item in text_layout_per_page if item == 'vertical')
    count_horizontal = sum(1 for item in text_layout_per_page if item == 'horizontal')
    known_layout_cnt = count_vertical + count_horizontal
    if known_layout_cnt == 0:
        return False
    ratio = count_vertical / known_layout_cnt
    return ratio < 0.5

def classify_by_img_narrow_strips(page_width, page_height, img_sz_list):
    """Classifie le PDF selon la présence de bandes d'images étroites.
    
    Args:
        page_width: Largeur de la page
        page_height: Hauteur de la page
        img_sz_list: Liste des tailles d'images par page
        
    Returns:
        True si peu de pages contiennent des bandes d'images étroites
    """
    def is_narrow_strip(img):
        x0, y0, x1, y1, _ = img
        width, height = x1 - x0, y1 - y0
        return any([
            width >= page_width * 0.9 and width >= height * 4,
            height >= page_height * 0.9 and height >= width * 4,
        ])

    narrow_strip_pages_count = 0
    for page_img_list in img_sz_list:
        if not page_img_list:
            continue

        total_images = len(page_img_list)
        narrow_strip_images_count = sum(1 for img in page_img_list if is_narrow_strip(img))
        if narrow_strip_images_count >= 5 and narrow_strip_images_count / total_images >= 0.8:
            narrow_strip_pages_count += 1

    narrow_strip_pages_ratio = narrow_strip_pages_count / len(img_sz_list)
    return narrow_strip_pages_ratio < 0.5

@staticmethod
def classify(total_page: int, page_width, page_height, img_sz_list: list, text_len_list: list, img_num_list: list,
             text_layout_list: list, invalid_chars: bool):
    """Fonction principale de classification des PDF.
    
    Combine plusieurs critères pour déterminer si un PDF est principalement du texte ou un scan.
    
    Args:
        total_page: Nombre total de pages
        page_width: Largeur de la page
        page_height: Hauteur de la page
        img_sz_list: Liste des tailles d'images par page
        text_len_list: Liste des longueurs de texte par page
        img_num_list: Liste du nombre d'images par page
        text_layout_list: Liste des orientations de texte par page
        invalid_chars: Présence de caractères invalides
        
    Returns:
        (bool, dict): True si PDF texte, False si scan, et dictionnaire des résultats par critère
    """
    results = {
        'by_image_area': classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list),
        'by_text_len': classify_by_text_len(text_len_list, total_page),
        'by_avg_words': classify_by_avg_words(text_len_list),
        'by_img_num': classify_by_img_num(img_sz_list, img_num_list),
        'by_text_layout': classify_by_text_layout(text_layout_list),
        'by_img_narrow_strips': classify_by_img_narrow_strips(page_width, page_height, img_sz_list),
        'by_invalid_chars': invalid_chars,
    }

    if all(results.values()):
        return True, results
    elif not any(results.values()):
        return False, results
    else:
        logger.warning(
            f"pdf is not classified by area and text_len, by_image_area: {results['by_image_area']},"
            f" by_text: {results['by_text_len']}, by_avg_words: {results['by_avg_words']}, by_img_num: {results['by_img_num']},"
            f" by_text_layout: {results['by_text_layout']}, by_img_narrow_strips: {results['by_img_narrow_strips']},"
            f" by_invalid_chars: {results['by_invalid_chars']}",
            file=sys.stderr)
        return False, results

def main(json_file):
    if json_file is None:
        print("json_file is None", file=sys.stderr)
        exit(0)
    try:
        with open(json_file, "r") as f:
            for l in f:
                if l.strip() == "":
                    continue
                o = json.loads(l)
                total_page = o["total_page"]
                page_width = o["page_width_pts"]
                page_height = o["page_height_pts"]
                img_sz_list = o["image_info_per_page"]
                text_len_list = o['text_len_per_page']
                text_layout_list = o['text_layout_per_page']
                pdf_path = o['pdf_path']
                is_encrypted = o['is_encrypted']
                is_needs_password = o['is_needs_password']
                if is_encrypted or total_page == 0 or is_needs_password:
                    continue
                tag = classify(total_page, page_width, page_height, img_sz_list, text_len_list, text_layout_list)
                o['is_text_pdf'] = tag
                print(json.dumps(o, ensure_ascii=False))
    except Exception as e:
        print("ERROR: ", e, file=sys.stderr)

if __name__ == "__main__":
    main()