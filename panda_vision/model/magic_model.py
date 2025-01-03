import enum

from panda_vision.config.model_block_type import ModelBlockTypeEnum
from panda_vision.config.ocr_content_type import CategoryId, ContentType
from panda_vision.data.dataset import Dataset
from panda_vision.libs.boxbase import (_is_in, _is_part_overlap, bbox_distance,
                                    bbox_relative_pos, box_area, calculate_iou,
                                    calculate_overlap_area_in_bbox1_area_ratio,
                                    get_overlap_area)
from panda_vision.libs.coordinate_transform import get_scale_ratio
from panda_vision.libs.local_math import float_gt
from panda_vision.pre_proc.remove_bbox_overlap import _remove_overlap_between_bbox

CAPATION_OVERLAP_AREA_RATIO = 0.6
MERGE_BOX_OVERLAP_AREA_RATIO = 1.1


class PosRelationEnum(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'
    UP = 'up'
    BOTTOM = 'bottom'
    ALL = 'all'


class MagicModel:
    """Retourne une liste vide si aucun élément n'est trouvé."""

    def __fix_axis(self):
        for model_page_info in self.__model_list:
            need_remove_list = []
            page_no = model_page_info['page_info']['page_no']
            horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(
                model_page_info, self.__docs.get_page(page_no)
            )
            layout_dets = model_page_info['layout_dets']
            for layout_det in layout_dets:

                if layout_det.get('bbox') is not None:
                    # Compatible avec les données de modèle qui produisent directement bbox, comme paddle
                    x0, y0, x1, y1 = layout_det['bbox']
                else:
                    # Compatible avec les données de modèle qui produisent directement poly, comme xxx
                    x0, y0, _, _, x1, y1, _, _ = layout_det['poly']

                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                layout_det['bbox'] = bbox
                # Supprimer les spans dont la hauteur ou la largeur est inférieure ou égale à 0
                if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    need_remove_list.append(layout_det)
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    def __fix_by_remove_low_confidence(self):
        for model_page_info in self.__model_list:
            need_remove_list = []
            layout_dets = model_page_info['layout_dets']
            for layout_det in layout_dets:
                if layout_det['score'] <= 0.05:
                    need_remove_list.append(layout_det)
                else:
                    continue
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    def __fix_by_remove_high_iou_and_low_confidence(self):
        for model_page_info in self.__model_list:
            need_remove_list = []
            layout_dets = model_page_info['layout_dets']
            for layout_det1 in layout_dets:
                for layout_det2 in layout_dets:
                    if layout_det1 == layout_det2:
                        continue
                    if layout_det1['category_id'] in [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                    ] and layout_det2['category_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        if (
                            calculate_iou(layout_det1['bbox'], layout_det2['bbox'])
                            > 0.9
                        ):
                            if layout_det1['score'] < layout_det2['score']:
                                layout_det_need_remove = layout_det1
                            else:
                                layout_det_need_remove = layout_det2

                            if layout_det_need_remove not in need_remove_list:
                                need_remove_list.append(layout_det_need_remove)
                        else:
                            continue
                    else:
                        continue
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)

    def __init__(self, model_list: list, docs: Dataset):
        self.__model_list = model_list
        self.__docs = docs
        """Ajouter des informations bbox pour toutes les données du modèle (mise à l'échelle, poly->bbox)"""
        self.__fix_axis()
        """Supprimer les données de modèle avec une confiance très faible (<0.05) pour améliorer la qualité"""
        self.__fix_by_remove_low_confidence()
        """Supprimer les données avec un iou élevé (>0.9) et une confiance plus faible"""
        self.__fix_by_remove_high_iou_and_low_confidence()
        self.__fix_footnote()

    def _bbox_distance(self, bbox1, bbox2):
        left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
        flags = [left, right, bottom, top]
        count = sum([1 if v else 0 for v in flags])
        if count > 1:
            return float('inf')
        if left or right:
            l1 = bbox1[3] - bbox1[1]
            l2 = bbox2[3] - bbox2[1]
        else:
            l1 = bbox1[2] - bbox1[0]
            l2 = bbox2[2] - bbox2[0]

        if l2 > l1 and (l2 - l1) / l1 > 0.3:
            return float('inf')

        return bbox_distance(bbox1, bbox2)

    def __fix_footnote(self):
        # 3: figure, 5: table, 7: footnote
        for model_page_info in self.__model_list:
            footnotes = []
            figures = []
            tables = []

            for obj in model_page_info['layout_dets']:
                if obj['category_id'] == 7:
                    footnotes.append(obj)
                elif obj['category_id'] == 3:
                    figures.append(obj)
                elif obj['category_id'] == 5:
                    tables.append(obj)
                if len(footnotes) * len(figures) == 0:
                    continue
            dis_figure_footnote = {}
            dis_table_footnote = {}

            for i in range(len(footnotes)):
                for j in range(len(figures)):
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], figures[j]['bbox']
                                ),
                            )
                        )
                    )
                    if pos_flag_count > 1:
                        continue
                    dis_figure_footnote[i] = min(
                        self._bbox_distance(figures[j]['bbox'], footnotes[i]['bbox']),
                        dis_figure_footnote.get(i, float('inf')),
                    )
            for i in range(len(footnotes)):
                for j in range(len(tables)):
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    footnotes[i]['bbox'], tables[j]['bbox']
                                ),
                            )
                        )
                    )
                    if pos_flag_count > 1:
                        continue

                    dis_table_footnote[i] = min(
                        self._bbox_distance(tables[j]['bbox'], footnotes[i]['bbox']),
                        dis_table_footnote.get(i, float('inf')),
                    )
            for i in range(len(footnotes)):
                if i not in dis_figure_footnote:
                    continue
                if dis_table_footnote.get(i, float('inf')) > dis_figure_footnote[i]:
                    footnotes[i]['category_id'] = CategoryId.ImageFootnote

    def __reduct_overlap(self, bboxes):
        N = len(bboxes)
        keep = [True] * N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if _is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                    keep[i] = False
        return [bboxes[i] for i in range(N) if keep[i]]

    def __tie_up_category_by_distance(
        self, page_no, subject_category_id, object_category_id
    ):
        """Suppose que chaque sujet a au maximum un objet (plusieurs objets adjacents peuvent être fusionnés en un seul objet), 
        chaque objet ne peut appartenir qu'à un seul sujet."""
        ret = []
        MAX_DIS_OF_POINT = 10**9 + 7
        """
        Les bbox du sujet et de l'objet seront fusionnés en une grande bbox (nommée: merged bbox).
        Filtrer tous les sujets qui ont un chevauchement avec la merged bbox et dont la surface de chevauchement est supérieure à la surface de l'objet.
        Puis calculer la distance minimale entre les sujets filtrés et l'objet
        """

        def search_overlap_between_boxes(subject_idx, object_idx):
            idxes = [subject_idx, object_idx]
            x0s = [all_bboxes[idx]['bbox'][0] for idx in idxes]
            y0s = [all_bboxes[idx]['bbox'][1] for idx in idxes]
            x1s = [all_bboxes[idx]['bbox'][2] for idx in idxes]
            y1s = [all_bboxes[idx]['bbox'][3] for idx in idxes]

            merged_bbox = [
                min(x0s),
                min(y0s),
                max(x1s),
                max(y1s),
            ]
            ratio = 0

            other_objects = list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id']
                        not in (object_category_id, subject_category_id),
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
            for other_object in other_objects:
                ratio = max(
                    ratio,
                    get_overlap_area(merged_bbox, other_object['bbox'])
                    * 1.0
                    / box_area(all_bboxes[object_idx]['bbox']),
                )
                if ratio >= MERGE_BOX_OVERLAP_AREA_RATIO:
                    break

            return ratio

        def may_find_other_nearest_bbox(subject_idx, object_idx):
            ret = float('inf')

            x0 = min(
                all_bboxes[subject_idx]['bbox'][0], all_bboxes[object_idx]['bbox'][0]
            )
            y0 = min(
                all_bboxes[subject_idx]['bbox'][1], all_bboxes[object_idx]['bbox'][1]
            )
            x1 = max(
                all_bboxes[subject_idx]['bbox'][2], all_bboxes[object_idx]['bbox'][2]
            )
            y1 = max(
                all_bboxes[subject_idx]['bbox'][3], all_bboxes[object_idx]['bbox'][3]
            )

            object_area = abs(
                all_bboxes[object_idx]['bbox'][2] - all_bboxes[object_idx]['bbox'][0]
            ) * abs(
                all_bboxes[object_idx]['bbox'][3] - all_bboxes[object_idx]['bbox'][1]
            )

            for i in range(len(all_bboxes)):
                if (
                    i == subject_idx
                    or all_bboxes[i]['category_id'] != subject_category_id
                ):
                    continue
                if _is_part_overlap([x0, y0, x1, y1], all_bboxes[i]['bbox']) or _is_in(
                    all_bboxes[i]['bbox'], [x0, y0, x1, y1]
                ):

                    i_area = abs(
                        all_bboxes[i]['bbox'][2] - all_bboxes[i]['bbox'][0]
                    ) * abs(all_bboxes[i]['bbox'][3] - all_bboxes[i]['bbox'][1])
                    if i_area >= object_area:
                        ret = min(float('inf'), dis[i][object_idx])

            return ret

        def expand_bbbox(idxes):
            x0s = [all_bboxes[idx]['bbox'][0] for idx in idxes]
            y0s = [all_bboxes[idx]['bbox'][1] for idx in idxes]
            x1s = [all_bboxes[idx]['bbox'][2] for idx in idxes]
            y1s = [all_bboxes[idx]['bbox'][3] for idx in idxes]
            return min(x0s), min(y0s), max(x1s), max(y1s)

        subjects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == subject_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )

        objects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == object_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )
        subject_object_relation_map = {}

        subjects.sort(
            key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2
        )  # get the distance !

        all_bboxes = []

        for v in subjects:
            all_bboxes.append(
                {
                    'category_id': subject_category_id,
                    'bbox': v['bbox'],
                    'score': v['score'],
                }
            )

        for v in objects:
            all_bboxes.append(
                {
                    'category_id': object_category_id,
                    'bbox': v['bbox'],
                    'score': v['score'],
                }
            )

        N = len(all_bboxes)
        dis = [[MAX_DIS_OF_POINT] * N for _ in range(N)]

        for i in range(N):
            for j in range(i):
                if (
                    all_bboxes[i]['category_id'] == subject_category_id
                    and all_bboxes[j]['category_id'] == subject_category_id
                ):
                    continue

                subject_idx, object_idx = i, j
                if all_bboxes[j]['category_id'] == subject_category_id:
                    subject_idx, object_idx = j, i

                if (
                    search_overlap_between_boxes(subject_idx, object_idx)
                    >= MERGE_BOX_OVERLAP_AREA_RATIO
                ):
                    dis[i][j] = float('inf')
                    dis[j][i] = dis[i][j]
                    continue

                dis[i][j] = self._bbox_distance(
                    all_bboxes[subject_idx]['bbox'], all_bboxes[object_idx]['bbox']
                )
                dis[j][i] = dis[i][j]

        used = set()
        for i in range(N):
            # chercher les objets associés au sujet i
            if all_bboxes[i]['category_id'] != subject_category_id:
                continue
            seen = set()
            candidates = []
            arr = []
            for j in range(N):

                pos_flag_count = sum(
                    list(
                        map(
                            lambda x: 1 if x else 0,
                            bbox_relative_pos(
                                all_bboxes[i]['bbox'], all_bboxes[j]['bbox']
                            ),
                        )
                    )
                )
                if pos_flag_count > 1:
                    continue
                if (
                    all_bboxes[j]['category_id'] != object_category_id
                    or j in used
                    or dis[i][j] == MAX_DIS_OF_POINT
                ):
                    continue
                left, right, _, _ = bbox_relative_pos(
                    all_bboxes[i]['bbox'], all_bboxes[j]['bbox']
                )  # la logique liée à pos_flag_count garantit l'exactitude de cette logique
                if left or right:
                    one_way_dis = all_bboxes[i]['bbox'][2] - all_bboxes[i]['bbox'][0]
                else:
                    one_way_dis = all_bboxes[i]['bbox'][3] - all_bboxes[i]['bbox'][1]
                if dis[i][j] > one_way_dis:
                    continue
                arr.append((dis[i][j], j))

            arr.sort(key=lambda x: x[0])
            if len(arr) > 0:
                """
                bug: l'objet le plus proche de ce sujet peut chevaucher d'autres sujets.
                Par exemple [ce sujet] [un autre sujet] [l'objet le plus proche du sujet]
                """
                if may_find_other_nearest_bbox(i, arr[0][1]) >= arr[0][0]:

                    candidates.append(arr[0][1])
                    seen.add(arr[0][1])

            # graine initiale obtenue
            for j in set(candidates):
                tmp = []
                for k in range(i + 1, N):
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    all_bboxes[j]['bbox'], all_bboxes[k]['bbox']
                                ),
                            )
                        )
                    )

                    if pos_flag_count > 1:
                        continue

                    if (
                        all_bboxes[k]['category_id'] != object_category_id
                        or k in used
                        or k in seen
                        or dis[j][k] == MAX_DIS_OF_POINT
                        or dis[j][k] > dis[i][j]
                    ):
                        continue

                    is_nearest = True
                    for ni in range(i + 1, N):
                        if ni in (j, k) or ni in used or ni in seen:
                            continue

                        if not float_gt(dis[ni][k], dis[j][k]):
                            is_nearest = False
                            break

                    if is_nearest:
                        nx0, ny0, nx1, ny1 = expand_bbbox(list(seen) + [k])
                        n_dis = bbox_distance(
                            all_bboxes[i]['bbox'], [nx0, ny0, nx1, ny1]
                        )
                        if float_gt(dis[i][j], n_dis):
                            continue
                        tmp.append(k)
                        seen.add(k)

                candidates = tmp
                if len(candidates) == 0:
                    break

            # On a déjà obtenu toutes les légendes les plus proches d'une figure donnée,
            # ainsi que les légendes les plus proches de ces légendes.
            # D'abord, élargissons la bbox
            ox0, oy0, ox1, oy1 = expand_bbbox(list(seen) + [i])
            ix0, iy0, ix1, iy1 = all_bboxes[i]['bbox']

            # L'espace est divisé en 4 zones de découpage, nous devons calculer 
            # la surface rectangulaire occupée par les objets fusionnés dans chaque zone
            caption_poses = [
                [ox0, oy0, ix0, oy1],
                [ox0, oy0, ox1, iy0], 
                [ox0, iy1, ox1, oy1],
                [ix1, oy0, ox1, oy1],
            ]

            caption_areas = []
            for bbox in caption_poses:
                embed_arr = []
                for idx in seen:
                    if (
                        calculate_overlap_area_in_bbox1_area_ratio(
                            all_bboxes[idx]['bbox'], bbox
                        )
                        > CAPATION_OVERLAP_AREA_RATIO
                    ):
                        embed_arr.append(idx)

                if len(embed_arr) > 0:
                    embed_x0 = min([all_bboxes[idx]['bbox'][0] for idx in embed_arr])
                    embed_y0 = min([all_bboxes[idx]['bbox'][1] for idx in embed_arr])
                    embed_x1 = max([all_bboxes[idx]['bbox'][2] for idx in embed_arr])
                    embed_y1 = max([all_bboxes[idx]['bbox'][3] for idx in embed_arr])
                    caption_areas.append(
                        int(abs(embed_x1 - embed_x0) * abs(embed_y1 - embed_y0))
                    )
                else:
                    caption_areas.append(0)

            subject_object_relation_map[i] = []
            if max(caption_areas) > 0:
                max_area_idx = caption_areas.index(max(caption_areas))
                caption_bbox = caption_poses[max_area_idx]

                for j in seen:
                    if (
                        calculate_overlap_area_in_bbox1_area_ratio(
                            all_bboxes[j]['bbox'], caption_bbox
                        )
                        > CAPATION_OVERLAP_AREA_RATIO
                    ):
                        used.add(j)
                        subject_object_relation_map[i].append(j)

        for i in sorted(subject_object_relation_map.keys()):
            result = {
                'subject_body': all_bboxes[i]['bbox'],
                'all': all_bboxes[i]['bbox'],
                'score': all_bboxes[i]['score'],
            }

            if len(subject_object_relation_map[i]) > 0:
                x0 = min(
                    [all_bboxes[j]['bbox'][0] for j in subject_object_relation_map[i]]
                )
                y0 = min(
                    [all_bboxes[j]['bbox'][1] for j in subject_object_relation_map[i]]
                )
                x1 = max(
                    [all_bboxes[j]['bbox'][2] for j in subject_object_relation_map[i]]
                )
                y1 = max(
                    [all_bboxes[j]['bbox'][3] for j in subject_object_relation_map[i]]
                )
                result['object_body'] = [x0, y0, x1, y1]
                result['all'] = [
                    min(x0, all_bboxes[i]['bbox'][0]),
                    min(y0, all_bboxes[i]['bbox'][1]),
                    max(x1, all_bboxes[i]['bbox'][2]),
                    max(y1, all_bboxes[i]['bbox'][3]),
                ]
            ret.append(result)

        total_subject_object_dis = 0
        # Calcule la distance entre les paires déjà associées
        for i in subject_object_relation_map.keys():
            for j in subject_object_relation_map[i]:
                total_subject_object_dis += bbox_distance(
                    all_bboxes[i]['bbox'], all_bboxes[j]['bbox']
                )

        # Calcule la distance entre les sujets et objets non appariés (version non précise)
        with_caption_subject = set(
            [
                key
                for key in subject_object_relation_map.keys()
                if len(subject_object_relation_map[i]) > 0
            ]
        )
        for i in range(N):
            if all_bboxes[i]['category_id'] != object_category_id or i in used:
                continue
            candidates = []
            for j in range(N):
                if (
                    all_bboxes[j]['category_id'] != subject_category_id
                    or j in with_caption_subject
                ):
                    continue
                candidates.append((dis[i][j], j))
            if len(candidates) > 0:
                candidates.sort(key=lambda x: x[0])
                total_subject_object_dis += candidates[0][1]
                with_caption_subject.add(j)
        return ret, total_subject_object_dis

    def __tie_up_category_by_distance_v2(
        self,
        page_no: int,
        subject_category_id: int,
        object_category_id: int,
        priority_pos: PosRelationEnum,
    ):
        """_summary_

        Args:
            page_no (int): _description_
            subject_category_id (int): _description_
            object_category_id (int): _description_
            priority_pos (PosRelationEnum): _description_

        Returns:
            _type_: _description_
        """
        AXIS_MULPLICITY = 0.5
        subjects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == subject_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )

        objects = self.__reduct_overlap(
            list(
                map(
                    lambda x: {'bbox': x['bbox'], 'score': x['score']},
                    filter(
                        lambda x: x['category_id'] == object_category_id,
                        self.__model_list[page_no]['layout_dets'],
                    ),
                )
            )
        )
        M = len(objects)

        subjects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)
        objects.sort(key=lambda x: x['bbox'][0] ** 2 + x['bbox'][1] ** 2)

        sub_obj_map_h = {i: [] for i in range(len(subjects))}

        dis_by_directions = {
            'top': [[-1, float('inf')]] * M,
            'bottom': [[-1, float('inf')]] * M,
            'left': [[-1, float('inf')]] * M,
            'right': [[-1, float('inf')]] * M,
        }

        for i, obj in enumerate(objects):
            l_x_axis, l_y_axis = (
                obj['bbox'][2] - obj['bbox'][0],
                obj['bbox'][3] - obj['bbox'][1],
            )
            axis_unit = min(l_x_axis, l_y_axis)
            for j, sub in enumerate(subjects):

                bbox1, bbox2, _ = _remove_overlap_between_bbox(
                    objects[i]['bbox'], subjects[j]['bbox']
                )
                left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)
                flags = [left, right, bottom, top]
                if sum([1 if v else 0 for v in flags]) > 1:
                    continue

                if left:
                    if dis_by_directions['left'][i][1] > bbox_distance(
                        obj['bbox'], sub['bbox']
                    ):
                        dis_by_directions['left'][i] = [
                            j,
                            bbox_distance(obj['bbox'], sub['bbox']),
                        ]
                if right:
                    if dis_by_directions['right'][i][1] > bbox_distance(
                        obj['bbox'], sub['bbox']
                    ):
                        dis_by_directions['right'][i] = [
                            j,
                            bbox_distance(obj['bbox'], sub['bbox']),
                        ]
                if bottom:
                    if dis_by_directions['bottom'][i][1] > bbox_distance(
                        obj['bbox'], sub['bbox']
                    ):
                        dis_by_directions['bottom'][i] = [
                            j,
                            bbox_distance(obj['bbox'], sub['bbox']),
                        ]
                if top:
                    if dis_by_directions['top'][i][1] > bbox_distance(
                        obj['bbox'], sub['bbox']
                    ):
                        dis_by_directions['top'][i] = [
                            j,
                            bbox_distance(obj['bbox'], sub['bbox']),
                        ]

            if (
                dis_by_directions['top'][i][1] != float('inf')
                and dis_by_directions['bottom'][i][1] != float('inf')
                and priority_pos in (PosRelationEnum.BOTTOM, PosRelationEnum.UP)
            ):
                RATIO = 3
                if (
                    abs(
                        dis_by_directions['top'][i][1]
                        - dis_by_directions['bottom'][i][1]
                    )
                    < RATIO * axis_unit
                ):

                    if priority_pos == PosRelationEnum.BOTTOM:
                        sub_obj_map_h[dis_by_directions['bottom'][i][0]].append(i)
                    else:
                        sub_obj_map_h[dis_by_directions['top'][i][0]].append(i)
                    continue

            if dis_by_directions['left'][i][1] != float('inf') or dis_by_directions[
                'right'
            ][i][1] != float('inf'):
                if dis_by_directions['left'][i][1] != float(
                    'inf'
                ) and dis_by_directions['right'][i][1] != float('inf'):
                    if AXIS_MULPLICITY * axis_unit >= abs(
                        dis_by_directions['left'][i][1]
                        - dis_by_directions['right'][i][1]
                    ):
                        left_sub_bbox = subjects[dis_by_directions['left'][i][0]][
                            'bbox'
                        ]
                        right_sub_bbox = subjects[dis_by_directions['right'][i][0]][
                            'bbox'
                        ]

                        left_sub_bbox_y_axis = left_sub_bbox[3] - left_sub_bbox[1]
                        right_sub_bbox_y_axis = right_sub_bbox[3] - right_sub_bbox[1]

                        if (
                            abs(left_sub_bbox_y_axis - l_y_axis)
                            + dis_by_directions['left'][i][0]
                            > abs(right_sub_bbox_y_axis - l_y_axis)
                            + dis_by_directions['right'][i][0]
                        ):
                            left_or_right = dis_by_directions['right'][i]
                        else:
                            left_or_right = dis_by_directions['left'][i]
                    else:
                        left_or_right = dis_by_directions['left'][i]
                        if left_or_right[1] > dis_by_directions['right'][i][1]:
                            left_or_right = dis_by_directions['right'][i]
                else:
                    left_or_right = dis_by_directions['left'][i]
                    if left_or_right[1] == float('inf'):
                        left_or_right = dis_by_directions['right'][i]
            else:
                left_or_right = [-1, float('inf')]

            if dis_by_directions['top'][i][1] != float('inf') or dis_by_directions[
                'bottom'
            ][i][1] != float('inf'):
                if dis_by_directions['top'][i][1] != float('inf') and dis_by_directions[
                    'bottom'
                ][i][1] != float('inf'):
                    if AXIS_MULPLICITY * axis_unit >= abs(
                        dis_by_directions['top'][i][1]
                        - dis_by_directions['bottom'][i][1]
                    ):
                        top_bottom = subjects[dis_by_directions['bottom'][i][0]]['bbox']
                        bottom_top = subjects[dis_by_directions['top'][i][0]]['bbox']

                        top_bottom_x_axis = top_bottom[2] - top_bottom[0]
                        bottom_top_x_axis = bottom_top[2] - bottom_top[0]
                        if (
                            abs(top_bottom_x_axis - l_x_axis)
                            + dis_by_directions['bottom'][i][1]
                            > abs(bottom_top_x_axis - l_x_axis)
                            + dis_by_directions['top'][i][1]
                        ):
                            top_or_bottom = dis_by_directions['top'][i]
                        else:
                            top_or_bottom = dis_by_directions['bottom'][i]
                    else:
                        top_or_bottom = dis_by_directions['top'][i]
                        if top_or_bottom[1] > dis_by_directions['bottom'][i][1]:
                            top_or_bottom = dis_by_directions['bottom'][i]
                else:
                    top_or_bottom = dis_by_directions['top'][i]
                    if top_or_bottom[1] == float('inf'):
                        top_or_bottom = dis_by_directions['bottom'][i]
            else:
                top_or_bottom = [-1, float('inf')]

            if left_or_right[1] != float('inf') or top_or_bottom[1] != float('inf'):
                if left_or_right[1] != float('inf') and top_or_bottom[1] != float(
                    'inf'
                ):
                    if AXIS_MULPLICITY * axis_unit >= abs(
                        left_or_right[1] - top_or_bottom[1]
                    ):
                        y_axis_bbox = subjects[left_or_right[0]]['bbox']
                        x_axis_bbox = subjects[top_or_bottom[0]]['bbox']

                        if (
                            abs((x_axis_bbox[2] - x_axis_bbox[0]) - l_x_axis) / l_x_axis
                            > abs((y_axis_bbox[3] - y_axis_bbox[1]) - l_y_axis)
                            / l_y_axis
                        ):
                            sub_obj_map_h[left_or_right[0]].append(i)
                        else:
                            sub_obj_map_h[top_or_bottom[0]].append(i)
                    else:
                        if left_or_right[1] > top_or_bottom[1]:
                            sub_obj_map_h[top_or_bottom[0]].append(i)
                        else:
                            sub_obj_map_h[left_or_right[0]].append(i)
                else:
                    if left_or_right[1] != float('inf'):
                        sub_obj_map_h[left_or_right[0]].append(i)
                    else:
                        sub_obj_map_h[top_or_bottom[0]].append(i)
        ret = []
        for i in sub_obj_map_h.keys():
            ret.append(
                {
                    'sub_bbox': {
                        'bbox': subjects[i]['bbox'],
                        'score': subjects[i]['score'],
                    },
                    'obj_bboxes': [
                        {'score': objects[j]['score'], 'bbox': objects[j]['bbox']}
                        for j in sub_obj_map_h[i]
                    ],
                    'sub_idx': i,
                }
            )
        return ret

    def get_imgs_v2(self, page_no: int):
        with_captions = self.__tie_up_category_by_distance_v2(
            page_no, 3, 4, PosRelationEnum.BOTTOM
        )
        with_footnotes = self.__tie_up_category_by_distance_v2(
            page_no, 3, CategoryId.ImageFootnote, PosRelationEnum.ALL
        )
        ret = []
        for v in with_captions:
            record = {
                'image_body': v['sub_bbox'],
                'image_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['image_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        return ret

    def get_tables_v2(self, page_no: int) -> list:
        with_captions = self.__tie_up_category_by_distance_v2(
            page_no, 5, 6, PosRelationEnum.UP
        )
        with_footnotes = self.__tie_up_category_by_distance_v2(
            page_no, 5, 7, PosRelationEnum.ALL
        )
        ret = []
        for v in with_captions:
            record = {
                'table_body': v['sub_bbox'],
                'table_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['table_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        return ret

    def get_imgs(self, page_no: int):
        with_captions, _ = self.__tie_up_category_by_distance(page_no, 3, 4)
        with_footnotes, _ = self.__tie_up_category_by_distance(
            page_no, 3, CategoryId.ImageFootnote
        )
        ret = []
        N, M = len(with_captions), len(with_footnotes)
        assert N == M
        for i in range(N):
            record = {
                'score': with_captions[i]['score'],
                'img_caption_bbox': with_captions[i].get('object_body', None),
                'img_body_bbox': with_captions[i]['subject_body'],
                'img_footnote_bbox': with_footnotes[i].get('object_body', None),
            }

            x0 = min(with_captions[i]['all'][0], with_footnotes[i]['all'][0])
            y0 = min(with_captions[i]['all'][1], with_footnotes[i]['all'][1])
            x1 = max(with_captions[i]['all'][2], with_footnotes[i]['all'][2])
            y1 = max(with_captions[i]['all'][3], with_footnotes[i]['all'][3])
            record['bbox'] = [x0, y0, x1, y1]
            ret.append(record)
        return ret

    def get_tables(
        self, page_no: int
    ) -> list:  # 3 coordonnées, légende, corps du tableau, note du tableau
        with_captions, _ = self.__tie_up_category_by_distance(page_no, 5, 6)
        with_footnotes, _ = self.__tie_up_category_by_distance(page_no, 5, 7)
        ret = []
        N, M = len(with_captions), len(with_footnotes)
        assert N == M
        for i in range(N):
            record = {
                'score': with_captions[i]['score'],
                'table_caption_bbox': with_captions[i].get('object_body', None),
                'table_body_bbox': with_captions[i]['subject_body'],
                'table_footnote_bbox': with_footnotes[i].get('object_body', None),
            }

            x0 = min(with_captions[i]['all'][0], with_footnotes[i]['all'][0])
            y0 = min(with_captions[i]['all'][1], with_footnotes[i]['all'][1])
            x1 = max(with_captions[i]['all'][2], with_footnotes[i]['all'][2])
            y1 = max(with_captions[i]['all'][3], with_footnotes[i]['all'][3])
            record['bbox'] = [x0, y0, x1, y1]
            ret.append(record)
        return ret

    def get_equations(self, page_no: int) -> list:  # a des coordonnées et du texte
        inline_equations = self.__get_blocks_by_type(
            ModelBlockTypeEnum.EMBEDDING.value, page_no, ['latex']
        )
        interline_equations = self.__get_blocks_by_type(
            ModelBlockTypeEnum.ISOLATED.value, page_no, ['latex']
        )
        interline_equations_blocks = self.__get_blocks_by_type(
            ModelBlockTypeEnum.ISOLATE_FORMULA.value, page_no
        )
        return inline_equations, interline_equations, interline_equations_blocks

    def get_discarded(self, page_no: int) -> list:  # modèle propriétaire, coordonnées uniquement
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.ABANDON.value, page_no)
        return blocks

    def get_text_blocks(self, page_no: int) -> list:  # fait par notre modèle, coordonnées uniquement, pas de texte
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.PLAIN_TEXT.value, page_no)
        return blocks

    def get_title_blocks(self, page_no: int) -> list:  # modèle propriétaire, coordonnées uniquement, pas de texte
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.TITLE.value, page_no)
        return blocks

    def get_ocr_text(self, page_no: int) -> list:  # fait par paddle, a du texte et des coordonnées
        text_spans = []
        model_page_info = self.__model_list[page_no]
        layout_dets = model_page_info['layout_dets']
        for layout_det in layout_dets:
            if layout_det['category_id'] == '15':
                span = {
                    'bbox': layout_det['bbox'],
                    'content': layout_det['text'],
                }
                text_spans.append(span)
        return text_spans

    def get_all_spans(self, page_no: int) -> list:

        def remove_duplicate_spans(spans):
            new_spans = []
            for span in spans:
                if not any(span == existing_span for existing_span in new_spans):
                    new_spans.append(span)
            return new_spans

        tous_spans = []
        info_page_modele = self.__model_list[page_no]
        dets_mise_en_page = info_page_modele['layout_dets']
        liste_categories_autorisees = [3, 5, 13, 14, 15]
        """Pour la concaténation des spans"""
        #  3: 'image', # Image
        #  5: 'table',       # Tableau
        #  13: 'inline_equation',     # Équation en ligne
        #  14: 'interline_equation',      # Équation entre les lignes
        #  15: 'text',      # Texte reconnu par OCR
        for det_mise_en_page in dets_mise_en_page:
            id_categorie = det_mise_en_page['category_id']
            if id_categorie in liste_categories_autorisees:
                span = {'bbox': det_mise_en_page['bbox'], 'score': det_mise_en_page['score']}
                if id_categorie == 3:
                    span['type'] = ContentType.Image
                elif id_categorie == 5:
                    # Obtenir les résultats du modèle de tableau
                    latex = det_mise_en_page.get('latex', None)
                    html = det_mise_en_page.get('html', None)
                    if latex:
                        span['latex'] = latex
                    elif html:
                        span['html'] = html
                    span['type'] = ContentType.Table
                elif id_categorie == 13:
                    span['content'] = det_mise_en_page['latex']
                    span['type'] = ContentType.InlineEquation
                elif id_categorie == 14:
                    span['content'] = det_mise_en_page['latex']
                    span['type'] = ContentType.InterlineEquation
                elif id_categorie == 15:
                    span['content'] = det_mise_en_page['text']
                    span['type'] = ContentType.Text
                tous_spans.append(span)
        return remove_duplicate_spans(tous_spans)

    def get_page_size(self, page_no: int):  # Obtenir la largeur et la hauteur de la page
        # Obtenir l'objet page pour la page courante
        page = self.__docs.get_page(page_no).get_page_info()
        # Obtenir la largeur et la hauteur de la page courante
        largeur_page = page.w
        hauteur_page = page.h
        return largeur_page, hauteur_page

    def __get_blocks_by_type(
        self, type: int, page_no: int, extra_col: list[str] = []
    ) -> list:
        blocks = []
        for page_dict in self.__model_list:
            layout_dets = page_dict.get('layout_dets', [])
            page_info = page_dict.get('page_info', {})
            page_number = page_info.get('page_no', -1)
            if page_no != page_number:
                continue
            for item in layout_dets:
                category_id = item.get('category_id', -1)
                bbox = item.get('bbox', None)

                if category_id == type:
                    block = {
                        'bbox': bbox,
                        'score': item.get('score'),
                    }
                    for col in extra_col:
                        block[col] = item.get(col, None)
                    blocks.append(block)
        return blocks

    def get_model_list(self, page_no):
        return self.__model_list[page_no]

