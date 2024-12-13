import copy
import os
import statistics
import time
from typing import List

import torch
import fitz
from loguru import logger

from panda_vision.config.enums import SupportedPdfParseMethod
from panda_vision.config.ocr_content_type import BlockType, ContentType
from panda_vision.data.dataset import Dataset, PageableData
from panda_vision.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from panda_vision.utils.clean_memory import clean_memory
from panda_vision.utils.config_reader import get_local_layoutreader_model_dir
from panda_vision.utils.convert_utils import dict_to_list
from panda_vision.utils.hash_utils import compute_md5

from panda_vision.utils.pdf_image_tools import cut_image_to_pil_image
from panda_vision.model.magic_model import MagicModel

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # désactiver la vérification des mises à jour albumentations
os.environ['YOLO_VERBOSE'] = 'False'  # désactiver le logger yolo

try:
    import torchtext

    if torchtext.__version__ >= "0.18.0":
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

from panda_vision.model.sub_modules.model_init import AtomModelSingleton
from panda_vision.para.para_split_v3 import para_split
from panda_vision.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from panda_vision.pre_proc.cut_image import ocr_cut_image_and_table
from panda_vision.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
from panda_vision.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
from panda_vision.pre_proc.ocr_span_list_modify import get_qa_need_list_v2, remove_overlaps_low_confidence_spans, remove_overlaps_min_spans


def __replace_STX_ETX(text_str: str):
    """Remplacer \u0002 et \u0003, car ces caractères deviennent illisibles lors de l'extraction avec pymupdf. En fait, c'étaient à l'origine des guillemets.

        Args:
            text_str (str): texte brut

        Returns:
            _type_: texte remplacé
    """
    if text_str:
        s = text_str.replace('\u0002', "'")
        s = s.replace('\u0003', "'")
        return s
    return text_str


def __replace_0xfffd(text_str: str):
    """Remplacer \ufffd, car ces caractères deviennent illisibles lors de l'extraction avec pymupdf."""
    if text_str:
        s = text_str.replace('\ufffd', " ")
        return s
    return text_str

def chars_to_content(span):
    # Vérifier si les caractères dans span sont vides
    if len(span['chars']) == 0:
        pass
        # span['content'] = ''
    else:
        # D'abord trier les chars selon la coordonnée x du centre de char['bbox']
        span['chars'] = sorted(span['chars'], key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)

        # Calculer la largeur moyenne des caractères
        char_width_sum = sum([char['bbox'][2] - char['bbox'][0] for char in span['chars']])
        char_avg_width = char_width_sum / len(span['chars'])

        content = ''
        for char in span['chars']:
            # Si la distance entre x0 du caractère suivant et x1 du caractère précédent dépasse la largeur d'un caractère, il faut insérer un espace entre les deux
            if char['bbox'][0] - span['chars'][span['chars'].index(char) - 1]['bbox'][2] > char_avg_width:
                content += ' '
            content += char['c']

        span['content'] = __replace_0xfffd(content)

    del span['chars']


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '"', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，', '-', '—', '–',)
LINE_START_FLAG = ('(', '（', '"', '"', '【', '{', '《', '<', '「', '『', '【', '[',)


def fill_char_in_spans(spans, all_chars):

    # Trier simplement du haut vers le bas
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    for char in all_chars:
        for span in spans:
            if calculate_char_in_span(char['bbox'], span['bbox'], char['c']):
                span['chars'].append(char)
                break

    empty_spans = []

    for span in spans:
        chars_to_content(span)
        # Certains spans n'ont pas de caractères mais ont un ou deux espaces réservés, filtrer par la largeur, la hauteur et la longueur du contenu
        if len(span['content']) * span['height'] < span['width'] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            empty_spans.append(span)
        del span['height'], span['width']
    return empty_spans


# Utiliser des coordonnées de point central plus robustes pour le jugement
def calculate_char_in_span(char_bbox, span_bbox, char, span_height_radio=0.33):
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        and abs(char_center_y - span_center_y) < span_height * span_height_radio  # La différence de hauteur entre l'axe central du caractère et l'axe central du span ne peut pas dépasser 1/4 de la hauteur du span
    ):
        return True
    else:
        # Si char est LINE_STOP_FLAG, ne pas utiliser le jugement du point central, utiliser une autre méthode (la frontière gauche est dans la zone span, le jugement de hauteur est cohérent avec la logique précédente)
        # Principalement pour donner une chance aux signes de ponctuation de fin d'entrer dans le span, ce char devrait également être proche de la frontière droite du span
        if char in LINE_STOP_FLAG:
            if (
                (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2]
                and char_center_x > span_bbox[0]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_radio
            ):
                return True
        elif char in LINE_START_FLAG:
            if (
                span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height)
                and char_center_x < span_bbox[2]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_radio
            ):
                return True
        else:
            return False


def txt_spans_extract_v2(pdf_page, spans, all_bboxes, all_discarded_blocks, lang):

    text_blocks_raw = pdf_page.get_text('rawdict', flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)['blocks']

    all_pymu_chars = []
    for block in text_blocks_raw:
        for line in block['lines']:
            cosine, sine = line['dir']
            if abs (cosine) < 0.9 or abs(sine) > 0.1:
                continue
            for span in line['spans']:
                all_pymu_chars.extend(span['chars'])

    # Calculer la médiane des hauteurs de tous les spans
    span_height_list = []
    for span in spans:
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        span_height = span['bbox'][3] - span['bbox'][1]
        span['height'] = span_height
        span['width'] = span['bbox'][2] - span['bbox'][0]
        span_height_list.append(span_height)
    if len(span_height_list) == 0:
        return spans
    else:
        median_span_height = statistics.median(span_height_list)

    useful_spans = []
    unuseful_spans = []
    # Deux caractéristiques des spans verticaux : 1. Hauteur dépassant plusieurs lignes 2. Rapport hauteur/largeur dépassant une certaine valeur
    vertical_spans = []
    for span in spans:
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        for block in all_bboxes + all_discarded_blocks:
            if block[7] in [BlockType.ImageBody, BlockType.TableBody, BlockType.InterlineEquation]:
                continue
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                if span['height'] > median_span_height * 3 and span['height'] > span['width'] * 3:
                    vertical_spans.append(span)
                elif block in all_bboxes:
                    useful_spans.append(span)
                else:
                    unuseful_spans.append(span)

                break

    """Remplir directement les spans verticaux avec les lignes pymu"""
    if len(vertical_spans) > 0:
        text_blocks = pdf_page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']
        all_pymu_lines = []
        for block in text_blocks:
            for line in block['lines']:
                all_pymu_lines.append(line)

        for pymu_line in all_pymu_lines:
            for span in vertical_spans:
                if calculate_overlap_area_in_bbox1_area_ratio(pymu_line['bbox'], span['bbox']) > 0.5:
                    for pymu_span in pymu_line['spans']:
                        span['content'] += pymu_span['text']
                    break

        for span in vertical_spans:
            if len(span['content']) == 0:
                spans.remove(span)

    """Si les spans horizontaux n'ont pas de caractères, utiliser l'OCR pour les remplir"""
    new_spans = []

    for span in useful_spans + unuseful_spans:
        if span['type'] in [ContentType.Text]:
            span['chars'] = []
            new_spans.append(span)

    empty_spans = fill_char_in_spans(new_spans, all_pymu_chars)

    if len(empty_spans) > 0:

        # Initialiser le modèle OCR
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name="ocr",
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang
        )

        for span in empty_spans:
            # Capturer l'image du span bbox puis OCR
            span_img = cut_image_to_pil_image(span['bbox'], pdf_page, mode="cv2")
            ocr_res = ocr_model.ocr(span_img, det=False)
            if ocr_res and len(ocr_res) > 0:
                if len(ocr_res[0]) > 0:
                    ocr_text, ocr_score = ocr_res[0][0]
                    # logger.info(f"ocr_text: {ocr_text}, ocr_score: {ocr_score}")
                    if ocr_score > 0.5 and len(ocr_text) > 0:
                        span['content'] = ocr_text
                        span['score'] = ocr_score
                    else:
                        spans.remove(span)

    return spans


def replace_text_span(pymu_spans, ocr_spans):
    return list(filter(lambda x: x['type'] != ContentType.Text, ocr_spans)) + pymu_spans


def model_init(model_name: str):
    from transformers import LayoutLMv3ForTokenClassification

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.is_bf16_supported():
            supports_bfloat16 = True
        else:
            supports_bfloat16 = False
    else:
        device = torch.device('cpu')
        supports_bfloat16 = False

    if model_name == 'layoutreader':
        # Vérifier si le répertoire de cache modelscope existe
        layoutreader_model_dir = get_local_layoutreader_model_dir()
        if os.path.exists(layoutreader_model_dir):
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                layoutreader_model_dir
            )
        else:
            logger.warning(
                'le modèle layoutreader local n\'existe pas, utiliser le modèle en ligne depuis huggingface'
            )
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                'hantian/layoutreader'
            )
        # Vérifier si l'appareil prend en charge bfloat16
        if supports_bfloat16:
            model.bfloat16()
        model.to(device).eval()
    else:
        logger.error('nom de modèle non autorisé')
        exit(1)
    return model


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name: str):
        if model_name not in self._models:
            self._models[model_name] = model_init(model_name=model_name)
        return self._models[model_name]


def do_predict(boxes: List[List[int]], model) -> List[int]:
    from panda_vision.model.sub_modules.reading_oreder.layoutreader.helpers import (
        boxes2inputs, parse_logits, prepare_inputs)

    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, model)
    logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))


def cal_block_index(fix_blocks, sorted_bboxes):

    if sorted_bboxes is not None:
        # Utiliser layoutreader pour trier
        for block in fix_blocks:
            line_index_list = []
            if len(block['lines']) == 0:
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                for line in block['lines']:
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    line_index_list.append(line['index'])
                median_value = statistics.median(line_index_list)
                block['index'] = median_value

            # Supprimer les informations de ligne virtuelle dans les blocs body d'image et de tableau, et remplir avec les informations real_lines
            if block['type'] in [BlockType.ImageBody, BlockType.TableBody]:
                block['virtual_lines'] = copy.deepcopy(block['lines'])
                block['lines'] = copy.deepcopy(block['real_lines'])
                del block['real_lines']
    else:
        # Utiliser xycut pour trier
        block_bboxes = []
        for block in fix_blocks:
            block_bboxes.append(block['bbox'])

            # Supprimer les informations de ligne virtuelle dans les blocs body d'image et de tableau, et remplir avec les informations real_lines
            if block['type'] in [BlockType.ImageBody, BlockType.TableBody]:
                block['virtual_lines'] = copy.deepcopy(block['lines'])
                block['lines'] = copy.deepcopy(block['real_lines'])
                del block['real_lines']

        import numpy as np

        from panda_vision.model.sub_modules.reading_oreder.layoutreader.xycut import \
            recursive_xy_cut

        random_boxes = np.array(block_bboxes)
        np.random.shuffle(random_boxes)
        res = []
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
        assert len(res) == len(block_bboxes)
        sorted_boxes = random_boxes[np.array(res)].tolist()

        for i, block in enumerate(fix_blocks):
            block['index'] = sorted_boxes.index(block['bbox'])

        # Générer l'index des lignes
        sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
        line_inedx = 1
        for block in sorted_blocks:
            for line in block['lines']:
                line['index'] = line_inedx
                line_inedx += 1

    return fix_blocks


def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    # block_bbox est un tuple (x0, y0, x1, y1), où (x0, y0) sont les coordonnées du coin inférieur gauche et (x1, y1) sont les coordonnées du coin supérieur droit
    x0, y0, x1, y1 = block_bbox

    block_height = y1 - y0
    block_weight = x1 - x0

    # Si la hauteur du bloc est inférieure à n lignes de texte, retourner directement le bbox du bloc
    if line_height * 3 < block_height:
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):  # Peut être une structure à deux colonnes, peut être coupée plus finement
            lines = int(block_height / line_height) + 1
        else:
            # Si la largeur du bloc dépasse 0.4 de la largeur de la page, diviser le bloc en 3 lignes (c'est une mise en page complexe, l'image ne peut pas être coupée trop finement)
            if block_weight > page_w * 0.4:
                line_height = (y1 - y0) / 3
                lines = 3
            elif block_weight > page_w * 0.25:  # (Peut être une structure à trois colonnes, aussi coupée plus finement)
                lines = int(block_height / line_height) + 1
            else:  # Juger le rapport hauteur/largeur
                if block_height / block_weight > 1.2:  # Ne pas diviser si c'est fin et long
                    return [[x0, y0, x1, y1]]
                else:  # Si ce n'est pas fin et long, diviser quand même en deux lignes
                    line_height = (y1 - y0) / 2
                    lines = 2

        # Déterminer à partir de quelle position y commencer à dessiner les lignes
        current_y = y0

        # Pour stocker les informations de position des lignes [(x0, y), ...]
        lines_positions = []

        for i in range(lines):
            lines_positions.append([x0, current_y, x1, current_y + line_height])
            current_y += line_height
        return lines_positions

    else:
        return [[x0, y0, x1, y1]]


def sort_lines_by_model(fix_blocks, page_w, page_h, line_height):
    page_line_list = []
    for block in fix_blocks:
        if block['type'] in [
            BlockType.Text, BlockType.Title, BlockType.InterlineEquation,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            if len(block['lines']) == 0:
                bbox = block['bbox']
                lines = insert_lines_into_block(bbox, line_height, page_w, page_h)
                for line in lines:
                    block['lines'].append({'bbox': line, 'spans': []})
                page_line_list.extend(lines)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in [BlockType.ImageBody, BlockType.TableBody]:
            bbox = block['bbox']
            block['real_lines'] = copy.deepcopy(block['lines'])
            lines = insert_lines_into_block(bbox, line_height, page_w, page_h)
            block['lines'] = []
            for line in lines:
                block['lines'].append({'bbox': line, 'spans': []})
            page_line_list.extend(lines)

    if len(page_line_list) > 200:  # layoutreader supporte au maximum 512 lignes
        return None

    # Utiliser layoutreader pour trier
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f'Boîte invalide. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
        boxes.append([left, top, right, bottom])
    model_manager = ModelSingleton()
    model = model_manager.get_model('layoutreader')
    with torch.no_grad():
        orders = do_predict(boxes, model)
    sorted_bboxes = [page_line_list[i] for i in orders]

    return sorted_bboxes


def get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block['type'] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            for line in block['lines']:
                bbox = line['bbox']
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10


def process_groups(groups, body_key, caption_key, footnote_key):
    body_blocks = []
    caption_blocks = []
    footnote_blocks = []
    for i, group in enumerate(groups):
        group[body_key]['group_id'] = i
        body_blocks.append(group[body_key])
        for caption_block in group[caption_key]:
            caption_block['group_id'] = i
            caption_blocks.append(caption_block)
        for footnote_block in group[footnote_key]:
            footnote_block['group_id'] = i
            footnote_blocks.append(footnote_block)
    return body_blocks, caption_blocks, footnote_blocks


def process_block_list(blocks, body_type, block_type):
    indices = [block['index'] for block in blocks]
    median_index = statistics.median(indices)

    body_bbox = next((block['bbox'] for block in blocks if block.get('type') == body_type), [])

    return {
        'type': block_type,
        'bbox': body_bbox,
        'blocks': blocks,
        'index': median_index,
    }


def revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block['type'] in [BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote]:
            group_id = block['group_id']
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block['type'] in [BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote]:
            group_id = block['group_id']
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.ImageBody, BlockType.Image))

    for group_id, blocks in table_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.TableBody, BlockType.Table))

    return new_blocks


def remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    def get_block_bboxes(blocks, block_type_list):
        return [block[0:4] for block in blocks if block[7] in block_type_list]

    image_bboxes = get_block_bboxes(all_bboxes, [BlockType.ImageBody])
    table_bboxes = get_block_bboxes(all_bboxes, [BlockType.TableBody])
    other_block_type = []
    for block_type in BlockType.__dict__.values():
        if not isinstance(block_type, str):
            continue
        if block_type not in [BlockType.ImageBody, BlockType.TableBody]:
            other_block_type.append(block_type)
    other_block_bboxes = get_block_bboxes(all_bboxes, other_block_type)
    discarded_block_bboxes = get_block_bboxes(all_discarded_blocks, [BlockType.Discarded])

    new_spans = []

    for span in spans:
        span_bbox = span['bbox']
        span_type = span['type']

        if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.4 for block_bbox in
               discarded_block_bboxes):
            new_spans.append(span)
            continue

        if span_type == ContentType.Image:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   image_bboxes):
                new_spans.append(span)
        elif span_type == ContentType.Table:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   table_bboxes):
                new_spans.append(span)
        else:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   other_block_bboxes):
                new_spans.append(span)

    return new_spans


def parse_page_core(
    page_doc: PageableData, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode, lang
):
    need_drop = False
    drop_reason = []

    """Obtenir les informations de bloc qui seront utilisées plus tard à partir de l'objet magic_model"""
    img_groups = magic_model.get_imgs_v2(page_id)
    table_groups = magic_model.get_tables_v2(page_id)

    """Grouper les blocs d'image et de tableau"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    discarded_blocks = magic_model.get_discarded(page_id)
    text_blocks = magic_model.get_text_blocks(page_id)
    title_blocks = magic_model.get_title_blocks(page_id)
    inline_equations, interline_equations, interline_equation_blocks = (
        magic_model.get_equations(page_id)
    )

    page_w, page_h = magic_model.get_page_size(page_id)

    """Rassembler tous les bbox des blocs"""
    # interline_equation_blocks n'est pas assez précis, on passe à interline_equations plus tard
    interline_equation_blocks = []
    if len(interline_equation_blocks) > 0:
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """Obtenir toutes les informations des spans"""
    spans = magic_model.get_all_spans(page_id)

    """Avant de supprimer les spans en double, il faut filtrer les spans d'image et de tableau via les blocs image_body et table_body"""
    """En même temps, supprimer les grands filigranes et conserver les spans abandonnés"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """Supprimer les spans qui se chevauchent avec une confiance plus faible"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """Supprimer les plus petits spans qui se chevauchent"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """Construire les spans selon le parse_mode, principalement pour le remplissage des caractères textuels"""
    if parse_mode == SupportedPdfParseMethod.TXT:

        """Utiliser la nouvelle version de la solution OCR hybride"""
        spans = txt_spans_extract_v2(page_doc, spans, all_bboxes, all_discarded_blocks, lang)

    elif parse_mode == SupportedPdfParseMethod.OCR:
        pass
    else:
        raise Exception('parse_mode must be txt or ocr')


    """Traiter d'abord les discarded_blocks qui ne nécessitent pas de mise en page"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """Si la page actuelle n'a pas de bbox valide, la sauter"""
    if len(all_bboxes) == 0:
        logger.warning(f'skip this page, not found useful bbox, page_id: {page_id}')
        return ocr_construct_page_component_v2(
            [],
            [],
            page_id,
            page_w,
            page_h,
            [],
            [],
            [],
            interline_equations,
            fix_discarded_blocks,
            need_drop,
            drop_reason,
        )

    """Capturer les images et tableaux"""
    spans = ocr_cut_image_and_table(
        spans, page_doc, page_id, pdf_bytes_md5, imageWriter
    )

    """Remplir les spans dans les blocs"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    """Effectuer l'opération de correction sur les blocs"""
    fix_blocks = fix_block_spans_v2(block_with_spans)

    """Obtenir toutes les lignes et calculer la hauteur des lignes de texte"""
    line_height = get_line_height(fix_blocks)

    """Obtenir toutes les lignes et les trier"""
    sorted_bboxes = sort_lines_by_model(fix_blocks, page_w, page_h, line_height)

    """Calculer les relations de séquence des blocs selon la médiane des lignes"""
    fix_blocks = cal_block_index(fix_blocks, sorted_bboxes)

    """Restaurer les blocs d'image et de tableau en forme de groupe pour le traitement ultérieur"""
    fix_blocks = revert_group_blocks(fix_blocks)

    """Réorganiser les blocs"""
    sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])

    """Obtenir les listes nécessaires pour QA"""
    images, tables, interline_equations = get_qa_need_list_v2(sorted_blocks)

    """Construire pdf_info_dict"""
    page_info = ocr_construct_page_component_v2(
        sorted_blocks,
        [],
        page_id,
        page_w,
        page_h,
        [],
        images,
        tables,
        interline_equations,
        fix_discarded_blocks,
        need_drop,
        drop_reason,
    )
    return page_info


def pdf_parse_union(
    dataset: Dataset,
    model_list,
    imageWriter,
    parse_mode,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
    lang=None,
):
    pdf_bytes_md5 = compute_md5(dataset.data_bits())

    """Initialiser un pdf_info_dict vide"""
    pdf_info_dict = {}

    """Initialiser magic_model avec model_list et l'objet docs"""
    magic_model = MagicModel(model_list, dataset)

    """Analyser le pdf selon la plage de départ spécifiée"""
    # end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    if end_page_id > len(dataset) - 1:
        logger.warning('end_page_id is out of range, use pdf_docs length')
        end_page_id = len(dataset) - 1

    """Initialiser le temps de démarrage"""
    start_time = time.time()

    for page_id, page in enumerate(dataset):
        """Afficher le temps d'analyse de chaque page en mode debug"""
        if debug_mode:
            time_now = time.time()
            logger.info(
                f'page_id: {page_id}, last_page_cost_time: {round(time.time() - start_time, 2)}'
            )
            start_time = time_now

        """Analyser chaque page du pdf"""
        if start_page_id <= page_id <= end_page_id:
            page_info = parse_page_core(
                page, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode, lang
            )
        else:
            page_info = page.get_page_info()
            page_w = page_info.w
            page_h = page_info.h
            page_info = ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [], [], [], True, 'skip page'
            )
        pdf_info_dict[f'page_{page_id}'] = page_info

    """Segmentation"""
    para_split(pdf_info_dict)

    """Conversion dict en list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        'pdf_info': pdf_info_list,
    }

    clean_memory()

    return new_pdf_info_dict


if __name__ == '__main__':
    pass
