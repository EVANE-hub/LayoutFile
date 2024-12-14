
from panda_vision.config.drop_tag import DropTag
from panda_vision.config.ocr_content_type import BlockType, ContentType
from panda_vision.utils.boxbase import (
    calculate_iou,
    get_minbox_if_overlap_by_ratio,
    calculate_overlap_area_in_bbox1_area_ratio,
    __is_overlaps_y_exceeds_threshold,
    calculate_vertical_projection_overlap_ratio,
)
from panda_vision.utils.commons import join_path
from panda_vision.utils.pdf_image_tools import cut_image
from loguru import logger

class OCRProcessor:
    def __init__(self):
        pass

    # Méthodes de ocr_span_list_modify.py
    def remove_overlaps_low_confidence_spans(self, spans):
        dropped_spans = []
        #  Supprimer les spans qui se chevauchent avec une confiance plus faible
        for span1 in spans:
            for span2 in spans:
                if span1 != span2:
                    # span1 ou span2 ne devraient pas être dans dropped_spans
                    if span1 in dropped_spans or span2 in dropped_spans:
                        continue
                    else:
                        if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                            if span1['score'] < span2['score']:
                                span_need_remove = span1
                            else:
                                span_need_remove = span2
                            if (
                                span_need_remove is not None
                                and span_need_remove not in dropped_spans
                            ):
                                dropped_spans.append(span_need_remove)

        if len(dropped_spans) > 0:
            for span_need_remove in dropped_spans:
                spans.remove(span_need_remove)
                span_need_remove['tag'] = DropTag.SPAN_OVERLAP

        return spans, dropped_spans

    def remove_overlaps_min_spans(self, spans):
        dropped_spans = []
        #  Supprimer les plus petits spans qui se chevauchent
        for span1 in spans:
            for span2 in spans:
                if span1 != span2:
                    # span1 ou span2 ne devraient pas être dans dropped_spans
                    if span1 in dropped_spans or span2 in dropped_spans:
                        continue
                    else:
                        overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                        if overlap_box is not None:
                            span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                            if span_need_remove is not None and span_need_remove not in dropped_spans:
                                dropped_spans.append(span_need_remove)
        if len(dropped_spans) > 0:
            for span_need_remove in dropped_spans:
                spans.remove(span_need_remove)
                span_need_remove['tag'] = DropTag.SPAN_OVERLAP

        return spans, dropped_spans

    def get_qa_need_list_v2(self, blocks):
        # Créer des copies de images, tables, interline_equations, inline_equations
        images = []
        tables = []
        interline_equations = []

        for block in blocks:
            if block['type'] == BlockType.Image:
                images.append(block)
            elif block['type'] == BlockType.Table:
                tables.append(block)
            elif block['type'] == BlockType.InterlineEquation:
                interline_equations.append(block)
        return images, tables, interline_equations

    # Méthodes de ocr_dict_merge.py
    def line_sort_spans_by_left_to_right(self, lines):
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

    def merge_spans_to_line(self, spans, threshold=0.6):
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

    def fill_spans_in_blocks(self, blocks, spans, radio):
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

    def fix_block_spans_v2(self, block_with_spans):
        fix_blocks = []
        for block in block_with_spans:
            block_type = block['type']

            if block_type in [BlockType.Text, BlockType.Title,
                            BlockType.ImageCaption, BlockType.ImageFootnote,
                            BlockType.TableCaption, BlockType.TableFootnote
                            ]:
                block = self.fix_text_block(block)
            elif block_type in [BlockType.InterlineEquation, BlockType.ImageBody, BlockType.TableBody]:
                block = self.fix_interline_block(block)
            else:
                continue
            fix_blocks.append(block)
        return fix_blocks

    def fix_discarded_block(self, discarded_block_with_spans):
        fix_discarded_blocks = []
        for block in discarded_block_with_spans:
            block = self.fix_text_block(block)
            fix_discarded_blocks.append(block)
        return fix_discarded_blocks

    def fix_text_block(self, block):
        # Les spans de formules dans un bloc de texte doivent être convertis en type inline
        for span in block['spans']:
            if span['type'] == ContentType.InterlineEquation:
                span['type'] = ContentType.InlineEquation
        block_lines = self.merge_spans_to_line(block['spans'])
        sort_block_lines = self.line_sort_spans_by_left_to_right(block_lines)
        block['lines'] = sort_block_lines
        del block['spans']
        return block

    def fix_interline_block(self, block):
        block_lines = self.merge_spans_to_line(block['spans'])
        sort_block_lines = self.line_sort_spans_by_left_to_right(block_lines)
        block['lines'] = sort_block_lines
        del block['spans']
        return block

    # Méthodes de ocr_detect_all_bboxes.py
    def add_bboxes(self, blocks, block_type, bboxes):
        for block in blocks:
            x0, y0, x1, y1 = block['bbox']
            if block_type in [
                BlockType.ImageBody,
                BlockType.ImageCaption,
                BlockType.ImageFootnote,
                BlockType.TableBody,
                BlockType.TableCaption,
                BlockType.TableFootnote,
            ]:
                bboxes.append(
                    [
                        x0,
                        y0,
                        x1,
                        y1,
                        None,
                        None,
                        None,
                        block_type,
                        None,
                        None,
                        None,
                        None,
                        block['score'],
                        block['group_id'],
                    ]
                )
            else:
                bboxes.append(
                    [
                        x0,
                        y0,
                        x1,
                        y1,
                        None,
                        None,
                        None,
                        block_type,
                        None,
                        None,
                        None,
                        None,
                        block['score'],
                    ]
                )

    def ocr_prepare_bboxes_for_layout_split_v2(self, img_body_blocks, img_caption_blocks, img_footnote_blocks, table_body_blocks, table_caption_blocks, table_footnote_blocks, discarded_blocks, text_blocks, title_blocks, interline_equation_blocks, page_w, page_h):
        all_bboxes = []

        self.add_bboxes(img_body_blocks, BlockType.ImageBody, all_bboxes)
        self.add_bboxes(img_caption_blocks, BlockType.ImageCaption, all_bboxes)
        self.add_bboxes(img_footnote_blocks, BlockType.ImageFootnote, all_bboxes)
        self.add_bboxes(table_body_blocks, BlockType.TableBody, all_bboxes)
        self.add_bboxes(table_caption_blocks, BlockType.TableCaption, all_bboxes)
        self.add_bboxes(table_footnote_blocks, BlockType.TableFootnote, all_bboxes)
        self.add_bboxes(text_blocks, BlockType.Text, all_bboxes)
        self.add_bboxes(title_blocks, BlockType.Title, all_bboxes)
        self.add_bboxes(interline_equation_blocks, BlockType.InterlineEquation, all_bboxes)

        """Résolution du problème d'imbrication des blocs"""
        """Quand le bloc de texte chevauche le bloc de titre, faire confiance au bloc de texte"""
        all_bboxes = self.fix_text_overlap_title_blocks(all_bboxes)
        """Quand un bloc chevauche un bloc rejeté, faire confiance au bloc rejeté"""
        all_bboxes = self.remove_need_drop_blocks(all_bboxes, discarded_blocks)

        # Conflit entre equation_interline et les blocs titre/texte - deux cas à traiter
        """Quand le bloc equation_interline a un IoU proche de 1 avec un bloc de texte, faire confiance au bloc d'équation"""
        all_bboxes = self.fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes)
        """Quand le bloc equation_interline est contenu dans un bloc de texte et beaucoup plus petit, faire confiance au bloc de texte et rejeter l'équation"""
        # Supprimé via la logique des grands blocs contenant les petits

        """blocs_rejetés"""
        all_discarded_blocks = []
        self.add_bboxes(discarded_blocks, BlockType.Discarded, all_discarded_blocks)

        """D��tection des notes de bas de page : largeur > 1/3 page, hauteur > 10, dans la moitié inférieure de la page"""
        footnote_blocks = []
        for discarded in discarded_blocks:
            x0, y0, x1, y1 = discarded['bbox']
            if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
                footnote_blocks.append([x0, y0, x1, y1])

        """Supprimer tous les blocs sous les notes de bas de page"""
        need_remove_blocks = self.find_blocks_under_footnote(all_bboxes, footnote_blocks)
        if len(need_remove_blocks) > 0:
            for block in need_remove_blocks:
                all_bboxes.remove(block)
                all_discarded_blocks.append(block)

        """Après ces traitements, s'il reste des grands blocs contenant des petits, supprimer les petits"""
        all_bboxes = self.remove_overlaps_min_blocks(all_bboxes)
        all_discarded_blocks = self.remove_overlaps_min_blocks(all_discarded_blocks)
        """Séparer les bbox restantes pour éviter les erreurs lors de la division du layout"""
        # all_bboxes, drop_reasons = remove_overlap_between_bbox_for_block(all_bboxes)
        all_bboxes.sort(key=lambda x: x[0]+x[1])
        return all_bboxes, all_discarded_blocks

    def find_blocks_under_footnote(self, all_bboxes, footnote_blocks):
        need_remove_blocks = []
        for block in all_bboxes:
            block_x0, block_y0, block_x1, block_y1 = block[:4]
            for footnote_bbox in footnote_blocks:
                footnote_x0, footnote_y0, footnote_x1, footnote_y1 = footnote_bbox
                # Si la projection verticale de la note couvre 80% de la projection du bloc et que y0 du bloc >= y1 de la note
                if (
                    block_y0 >= footnote_y1
                    and calculate_vertical_projection_overlap_ratio(
                        (block_x0, block_y0, block_x1, block_y1), footnote_bbox
                    )
                    >= 0.8
                ):
                    if block not in need_remove_blocks:
                        need_remove_blocks.append(block)
                        break
        return need_remove_blocks

    def fix_interline_equation_overlap_text_blocks_with_hi_iou(self, all_bboxes):
        # Extraire d'abord tous les blocs texte et interline
        text_blocks = []
        for block in all_bboxes:
            if block[7] == BlockType.Text:
                text_blocks.append(block)
        interline_equation_blocks = []
        for block in all_bboxes:
            if block[7] == BlockType.InterlineEquation:
                interline_equation_blocks.append(block)

        need_remove = []

        for interline_equation_block in interline_equation_blocks:
            for text_block in text_blocks:
                interline_equation_block_bbox = interline_equation_block[:4]
                text_block_bbox = text_block[:4]
                if calculate_iou(interline_equation_block_bbox, text_block_bbox) > 0.8:
                    if text_block not in need_remove:
                        need_remove.append(text_block)

        if len(need_remove) > 0:
            for block in need_remove:
                all_bboxes.remove(block)

        return all_bboxes

    def fix_text_overlap_title_blocks(self, all_bboxes):
        # Extraire d'abord tous les blocs texte et titre
        text_blocks = []
        for block in all_bboxes:
            if block[7] == BlockType.Text:
                text_blocks.append(block)
        title_blocks = []
        for block in all_bboxes:
            if block[7] == BlockType.Title:
                title_blocks.append(block)

        need_remove = []

        for text_block in text_blocks:
            for title_block in title_blocks:
                text_block_bbox = text_block[:4]
                title_block_bbox = title_block[:4]
                if calculate_iou(text_block_bbox, title_block_bbox) > 0.8:
                    if title_block not in need_remove:
                        need_remove.append(title_block)

        if len(need_remove) > 0:
            for block in need_remove:
                all_bboxes.remove(block)

        return all_bboxes

    def remove_need_drop_blocks(self, all_bboxes, discarded_blocks):
        need_remove = []
        for block in all_bboxes:
            for discarded_block in discarded_blocks:
                block_bbox = block[:4]
                if (
                    calculate_overlap_area_in_bbox1_area_ratio(
                        block_bbox, discarded_block['bbox']
                    )
                    > 0.6
                ):
                    if block not in need_remove:
                        need_remove.append(block)
                        break

        if len(need_remove) > 0:
            for block in need_remove:
                all_bboxes.remove(block)
        return all_bboxes

    def remove_overlaps_min_blocks(self, all_bboxes):
        # Les blocs qui se chevauchent ne peuvent pas être simplement supprimés, ils doivent être fusionnés avec le plus grand
        # Supprimer les plus petits blocs qui se chevauchent
        need_remove = []
        for block1 in all_bboxes:
            for block2 in all_bboxes:
                if block1 != block2:
                    block1_bbox = block1[:4]
                    block2_bbox = block2[:4]
                    overlap_box = get_minbox_if_overlap_by_ratio(
                        block1_bbox, block2_bbox, 0.8
                    )
                    if overlap_box is not None:
                        block_to_remove = next(
                            (block for block in all_bboxes if block[:4] == overlap_box),
                            None,
                        )
                        if (
                            block_to_remove is not None
                            and block_to_remove not in need_remove
                        ):
                            large_block = block1 if block1 != block_to_remove else block2
                            x1, y1, x2, y2 = large_block[:4]
                            sx1, sy1, sx2, sy2 = block_to_remove[:4]
                            x1 = min(x1, sx1)
                            y1 = min(y1, sy1)
                            x2 = max(x2, sx2)
                            y2 = max(y2, sy2)
                            large_block[:4] = [x1, y1, x2, y2]
                            need_remove.append(block_to_remove)

        if len(need_remove) > 0:
            for block in need_remove:
                all_bboxes.remove(block)

        return all_bboxes

    # Méthodes de cut_image.py
    def ocr_cut_image_and_table(self, spans, page, page_id, pdf_bytes_md5, imageWriter):
        def return_path(type):
            return join_path(pdf_bytes_md5, type)

        for span in spans:
            span_type = span['type']
            if span_type == ContentType.Image:
                if not self.check_img_bbox(span['bbox']) or not imageWriter:
                    continue
                span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('images'),
                                            imageWriter=imageWriter)
            elif span_type == ContentType.Table:
                if not self.check_img_bbox(span['bbox']) or not imageWriter:
                    continue
                span['image_path'] = cut_image(span['bbox'], page_id, page, return_path=return_path('tables'),
                                            imageWriter=imageWriter)

        return spans

    def check_img_bbox(self, bbox) -> bool:
        if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
            logger.warning(f'boîtes d\'images: boîte invalide, {bbox}')
            return False
        return True