from panda_vision.config.ocr_content_type import BlockType, ContentType
from panda_vision.utils.boxbase import __is_overlaps_y_exceeds_threshold, calculate_overlap_area_in_bbox1_area_ratio


# Trier les spans dans chaque ligne de gauche à droite
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        #  Trier par coordonnée x0
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def merge_spans_to_line(spans, threshold=0.6):
    if len(spans) == 0:
        return []
    else:
        # Trier par coordonnée y0
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # Si le type du span actuel est "interline_equation" ou s'il y a déjà un "interline_equation" dans la ligne actuelle
            # Pour les types image et table, même logique
            if span['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] or any(s['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] for s in current_line):
                # Commencer une nouvelle ligne
                lines.append(current_line)
                current_line = [span]
                continue

            # Si le span actuel chevauche le dernier span de la ligne actuelle sur l'axe y, l'ajouter à la ligne actuelle
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # Sinon, commencer une nouvelle ligne
                lines.append(current_line)
                current_line = [span]

        # Ajouter la dernière ligne
        if current_line:
            lines.append(current_line)

        return lines


def fill_spans_in_blocks(blocks, spans, radio):
    """Placer les spans de allspans dans les blocks selon leurs relations de position."""
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        if block_type in [
            BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote
        ]:
            block_dict['group_id'] = block[-1]
        block_spans = []
        for span in spans:
            span_bbox = span['bbox']
            if calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox, block_bbox) > radio:
                block_spans.append(span)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # Supprimer les spans déjà placés dans block_spans
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def fix_block_spans_v2(block_with_spans):
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type in [BlockType.Text, BlockType.Title,
                          BlockType.ImageCaption, BlockType.ImageFootnote,
                          BlockType.TableCaption, BlockType.TableFootnote
                          ]:
            block = fix_text_block(block)
        elif block_type in [BlockType.InterlineEquation, BlockType.ImageBody, BlockType.TableBody]:
            block = fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def fix_discarded_block(discarded_block_with_spans):
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks


def fix_text_block(block):
    # Les spans de formules dans un bloc de texte doivent être convertis en type inline
    for span in block['spans']:
        if span['type'] == ContentType.InterlineEquation:
            span['type'] = ContentType.InlineEquation
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def fix_interline_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block
