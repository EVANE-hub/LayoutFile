import copy

from loguru import logger

from panda_vision.config.constants import CROSS_PAGE, LINES_DELETED
from panda_vision.config.ocr_content_type import BlockType, ContentType
from panda_vision.utils.language import detect_lang

LINE_STOP_FLAG = (
    '.',
    '!',
    '?',
    '。',
    '！',
    '？',
    ')',
    '）',
    '"',
    '"',
    ':',
    '：',
    ';',
    '；',
)
LIST_END_FLAG = ('.', '。', ';', '；')


class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'


def __process_blocks(blocks):
    # Prétraitement de tous les blocs
    # 1. Grouper les blocs par titre et équation interlinéaire
    # 2. Réinitialiser les limites de bbox selon les informations de ligne

    result = []
    current_group = []

    for i in range(len(blocks)):
        current_block = blocks[i]

        # Si le bloc actuel est de type 'text'
        if current_block['type'] == 'text':
            current_block['bbox_fs'] = copy.deepcopy(current_block['bbox'])
            if 'lines' in current_block and len(current_block['lines']) > 0:
                current_block['bbox_fs'] = [
                    min([line['bbox'][0] for line in current_block['lines']]),
                    min([line['bbox'][1] for line in current_block['lines']]),
                    max([line['bbox'][2] for line in current_block['lines']]),
                    max([line['bbox'][3] for line in current_block['lines']]),
                ]
            current_group.append(current_block)

        # Vérifier si le bloc suivant existe
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            # Si le bloc suivant n'est pas de type 'text' et est de type 'title' ou 'interline_equation'
            if next_block['type'] in ['title', 'interline_equation']:
                result.append(current_group)
                current_group = []

    # Traiter le dernier groupe
    if current_group:
        result.append(current_group)

    return result


def __is_list_or_index_block(block):
    # Un bloc est un bloc de liste s'il remplit les conditions suivantes:
    # 1. Le bloc contient plusieurs lignes 2. Plusieurs lignes sont alignées à gauche 3. Plusieurs lignes ne sont pas alignées à droite (en dents de scie)
    # 1. Le bloc contient plusieurs lignes 2. Plusieurs lignes sont alignées à gauche 3. Plusieurs lignes se terminent par un drapeau de fin
    # 1. Le bloc contient plusieurs lignes 2. Plusieurs lignes sont alignées à gauche 3. Plusieurs lignes ne sont pas alignées à gauche

    # Un bloc d'index est un type spécial de bloc de liste
    # Un bloc est un bloc d'index s'il remplit les conditions suivantes:
    # 1. Le bloc contient plusieurs lignes 2. Les lignes sont alignées des deux côtés 3. Les lignes commencent ou se terminent par des chiffres
    if len(block['lines']) >= 2:
        first_line = block['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block_weight = block['bbox_fs'][2] - block['bbox_fs'][0]
        block_height = block['bbox_fs'][3] - block['bbox_fs'][1]
        page_weight, page_height = block['page_size']

        left_close_num = 0
        left_not_close_num = 0
        right_not_close_num = 0
        right_close_num = 0
        lines_text_list = []
        center_close_num = 0
        external_sides_not_close_num = 0
        multiple_para_flag = False
        last_line = block['lines'][-1]

        if page_weight == 0:
            block_weight_radio = 0
        else:
            block_weight_radio = block_weight / page_weight
        # logger.info(f"block_weight_radio: {block_weight_radio}")

        # Si la première ligne n'est pas alignée à gauche mais alignée à droite, et la dernière ligne est alignée à gauche mais pas à droite
        if (
            first_line['bbox'][0] - block['bbox_fs'][0] > line_height / 2
            and abs(last_line['bbox'][0] - block['bbox_fs'][0]) < line_height / 2
            and block['bbox_fs'][2] - last_line['bbox'][2] > line_height
        ):
            multiple_para_flag = True

        for line in block['lines']:
            line_mid_x = (line['bbox'][0] + line['bbox'][2]) / 2
            block_mid_x = (block['bbox_fs'][0] + block['bbox_fs'][2]) / 2
            if (
                line['bbox'][0] - block['bbox_fs'][0] > 0.7 * line_height
                and block['bbox_fs'][2] - line['bbox'][2] > 0.7 * line_height
            ):
                external_sides_not_close_num += 1
            if abs(line_mid_x - block_mid_x) < line_height / 2:
                center_close_num += 1

            line_text = ''

            for span in line['spans']:
                span_type = span['type']
                if span_type == ContentType.Text:
                    line_text += span['content'].strip()

            # Ajouter tout le texte, y compris les lignes vides, pour maintenir la cohérence avec la longueur de block['lines']
            lines_text_list.append(line_text)
            block_text = ''.join(lines_text_list)
            block_lang = detect_lang(block_text)
            # logger.info(f"block_lang: {block_lang}")

            # Calculer si le nombre de lignes alignées à gauche est supérieur à 2
            if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                left_close_num += 1
            elif line['bbox'][0] - block['bbox_fs'][0] > line_height:
                left_not_close_num += 1

            # Vérifier l'alignement à droite
            if abs(block['bbox_fs'][2] - line['bbox'][2]) < line_height:
                right_close_num += 1
            else:
                # Pour les langues comme le chinois sans mots longs, utiliser un seuil uniforme
                if block_lang in ['zh', 'ja', 'ko']:
                    closed_area = 0.26 * block_weight
                else:
                    # Seuil pour le non-alignement à droite
                    if block_weight_radio >= 0.5:
                        closed_area = 0.26 * block_weight
                    else:
                        closed_area = 0.36 * block_weight
                if block['bbox_fs'][2] - line['bbox'][2] > closed_area:
                    right_not_close_num += 1

        # Vérifier si plus de 80% des éléments se terminent par LIST_END_FLAG
        line_end_flag = False
        # Vérifier si plus de 80% des éléments commencent ou se terminent par un chiffre
        line_num_flag = False
        num_start_count = 0
        num_end_count = 0
        flag_end_count = 0

        if len(lines_text_list) > 0:
            for line_text in lines_text_list:
                if len(line_text) > 0:
                    if line_text[-1] in LIST_END_FLAG:
                        flag_end_count += 1
                    if line_text[0].isdigit():
                        num_start_count += 1
                    if line_text[-1].isdigit():
                        num_end_count += 1

            if (
                num_start_count / len(lines_text_list) >= 0.8
                or num_end_count / len(lines_text_list) >= 0.8
            ):
                line_num_flag = True
            if flag_end_count / len(lines_text_list) >= 0.8:
                line_end_flag = True

        # Certains sommaires ne sont pas alignés à droite
        if (
            left_close_num / len(block['lines']) >= 0.8
            or right_close_num / len(block['lines']) >= 0.8
        ) and line_num_flag:
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.Index

        # Reconnaissance spéciale des listes centrées
        elif (
            external_sides_not_close_num >= 2
            and center_close_num == len(block['lines'])
            and external_sides_not_close_num / len(block['lines']) >= 0.5
            and block_height / block_weight > 0.4
        ):
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.List

        elif (
            left_close_num >= 2
            and (right_not_close_num >= 2 or line_end_flag or left_not_close_num >= 2)
            and not multiple_para_flag
            # and block_weight_radio > 0.27
        ):
            # Traitement des listes sans indentation
            if left_close_num / len(block['lines']) > 0.8:
                # Liste d'éléments courts d'une seule ligne alignés à gauche
                if flag_end_count == 0 and right_close_num / len(block['lines']) < 0.5:
                    for line in block['lines']:
                        if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                # Cas où la plupart des éléments ont un marqueur de fin
                elif line_end_flag:
                    for i, line in enumerate(block['lines']):
                        if (
                            len(lines_text_list[i]) > 0
                            and lines_text_list[i][-1] in LIST_END_FLAG
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            if i + 1 < len(block['lines']):
                                block['lines'][i + 1][
                                    ListLineTag.IS_LIST_START_LINE
                                ] = True
                # Éléments sans marqueur de fin ni indentation
                else:
                    line_start_flag = False
                    for i, line in enumerate(block['lines']):
                        if line_start_flag:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                            line_start_flag = False

                        if (
                            abs(block['bbox_fs'][2] - line['bbox'][2])
                            > 0.1 * block_weight
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            line_start_flag = True
            # Liste ordonnée spéciale avec indentation
            elif num_start_count >= 2 and num_start_count == flag_end_count:
                for i, line in enumerate(block['lines']):
                    if len(lines_text_list[i]) > 0:
                        if lines_text_list[i][0].isdigit():
                            line[ListLineTag.IS_LIST_START_LINE] = True
                        if lines_text_list[i][-1] in LIST_END_FLAG:
                            line[ListLineTag.IS_LIST_END_LINE] = True
            else:
                # Traitement normal des listes avec indentation
                for line in block['lines']:
                    if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                        line[ListLineTag.IS_LIST_START_LINE] = True
                    if abs(block['bbox_fs'][2] - line['bbox'][2]) > line_height:
                        line[ListLineTag.IS_LIST_END_LINE] = True

            return BlockType.List
        else:
            return BlockType.Text
    else:
        return BlockType.Text


def __merge_2_text_blocks(block1, block2):
    if len(block1['lines']) > 0:
        first_line = block1['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block1_weight = block1['bbox'][2] - block1['bbox'][0]
        block2_weight = block2['bbox'][2] - block2['bbox'][0]
        min_block_weight = min(block1_weight, block2_weight)
        if abs(block1['bbox_fs'][0] - first_line['bbox'][0]) < line_height / 2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_height = last_line['bbox'][3] - last_line['bbox'][1]
                if len(first_line['spans']) > 0:
                    first_span = first_line['spans'][0]
                    if len(first_span['content']) > 0:
                        span_start_with_num = first_span['content'][0].isdigit()
                        span_start_with_big_char = first_span['content'][0].isupper()
                        if (
                            # La différence entre la limite droite de la dernière ligne et la limite droite du bloc ne dépasse pas line_height
                            abs(block2['bbox_fs'][2] - last_line['bbox'][2]) < line_height
                            # Le dernier span ne se termine pas par un symbole spécifique
                            and not last_span['content'].endswith(LINE_STOP_FLAG)
                            # La différence de largeur entre les deux blocs ne dépasse pas le double
                            and abs(block1_weight - block2_weight) < min_block_weight
                            # Le premier caractère du bloc suivant n'est pas un chiffre
                            and not span_start_with_num
                            # Le premier caractère du bloc suivant n'est pas une majuscule
                            and not span_start_with_big_char
                        ):
                            if block1['page_num'] != block2['page_num']:
                                for line in block1['lines']:
                                    for span in line['spans']:
                                        span[CROSS_PAGE] = True
                            block2['lines'].extend(block1['lines'])
                            block1['lines'] = []
                            block1[LINES_DELETED] = True

    return block1, block2


def __merge_2_list_blocks(block1, block2):
    if block1['page_num'] != block2['page_num']:
        for line in block1['lines']:
            for span in line['spans']:
                span[CROSS_PAGE] = True
    block2['lines'].extend(block1['lines'])
    block1['lines'] = []
    block1[LINES_DELETED] = True

    return block1, block2


def __is_list_group(text_blocks_group):
    # Les caractéristiques d'un groupe de liste sont que tous les blocs du groupe remplissent les conditions suivantes:
    # 1. Chaque bloc ne dépasse pas 3 lignes 2. Les limites gauches de chaque bloc sont relativement proches (règle simplifiée pour l'instant)
    for block in text_blocks_group:
        if len(block['lines']) > 3:
            return False
    return True


def __para_merge_page(blocks):
    page_text_blocks_groups = __process_blocks(blocks)
    for text_blocks_group in page_text_blocks_groups:
        if len(text_blocks_group) > 0:
            # Il faut d'abord déterminer si chaque bloc est un bloc de liste ou d'index avant la fusion
            for block in text_blocks_group:
                block_type = __is_list_or_index_block(block)
                block['type'] = block_type
                # logger.info(f"{block['type']}:{block}")

        if len(text_blocks_group) > 1:
            # Déterminer si ce groupe est un groupe de liste avant la fusion
            is_list_group = __is_list_group(text_blocks_group)

            # Parcours en ordre inverse
            for i in range(len(text_blocks_group) - 1, -1, -1):
                current_block = text_blocks_group[i]

                # Vérifier s'il y a un bloc précédent
                if i - 1 >= 0:
                    prev_block = text_blocks_group[i - 1]

                    if (
                        current_block['type'] == 'text'
                        and prev_block['type'] == 'text'
                        and not is_list_group
                    ):
                        __merge_2_text_blocks(current_block, prev_block)
                    elif (
                        current_block['type'] == BlockType.List
                        and prev_block['type'] == BlockType.List
                    ) or (
                        current_block['type'] == BlockType.Index
                        and prev_block['type'] == BlockType.Index
                    ):
                        __merge_2_list_blocks(current_block, prev_block)

        else:
            continue


def para_split(pdf_info_dict):
    all_blocks = []
    for page_num, page in pdf_info_dict.items():
        blocks = copy.deepcopy(page['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_num
            block['page_size'] = page['page_size']
        all_blocks.extend(blocks)

    __para_merge_page(all_blocks)
    for page_num, page in pdf_info_dict.items():
        page['para_blocks'] = []
        for block in all_blocks:
            if block['page_num'] == page_num:
                page['para_blocks'].append(block)


if __name__ == '__main__':
    input_blocks = []
    # Appeler la fonction
    groups = __process_blocks(input_blocks)
    for group_index, group in enumerate(groups):
        print(f'Group {group_index}: {group}')