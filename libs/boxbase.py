import math


def _is_in_or_part_overlap(box1, box2) -> bool:
    """Vérifie si deux bbox se chevauchent partiellement ou sont incluses l'une dans l'autre."""
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return not (x1_1 < x0_2 or  # box1 est à gauche de box2
                x0_1 > x1_2 or  # box1 est à droite de box2 
                y1_1 < y0_2 or  # box1 est au-dessus de box2
                y0_1 > y1_2)  # box1 est en-dessous de box2


def _is_in_or_part_overlap_with_area_ratio(box1,
                                           box2,
                                           area_ratio_threshold=0.6):
    """Vérifie si box1 est dans box2, ou si box1 et box2 se chevauchent partiellement avec un ratio de surface supérieur à area_ratio_threshold."""
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    if not _is_in_or_part_overlap(box1, box2):
        return False

    # Calcul de la surface de chevauchement
    x_left = max(x0_1, x0_2)
    y_top = max(y0_1, y0_2)
    x_right = min(x1_1, x1_2)
    y_bottom = min(y1_1, y1_2)
    overlap_area = (x_right - x_left) * (y_bottom - y_top)

    # Calcul de la surface de box1
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)

    return overlap_area / box1_area > area_ratio_threshold


def _is_in(box1, box2) -> bool:
    """Vérifie si box1 est complètement incluse dans box2."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (x0_1 >= x0_2 and  # limite gauche de box1 n'est pas à l'extérieur gauche de box2
            y0_1 >= y0_2 and  # limite supérieure de box1 n'est pas à l'extérieur supérieur de box2
            x1_1 <= x1_2 and  # limite droite de box1 n'est pas à l'extérieur droit de box2
            y1_1 <= y1_2)  # limite inférieure de box1 n'est pas à l'extérieur inférieur de box2


def _is_part_overlap(box1, box2) -> bool:
    """Vérifie si deux bbox se chevauchent partiellement, sans inclusion totale."""
    if box1 is None or box2 is None:
        return False

    return _is_in_or_part_overlap(box1, box2) and not _is_in(box1, box2)


def _left_intersect(left_box, right_box):
    """Vérifie si les limites gauches des deux box s'intersectent, c'est-à-dire si la limite droite de left_box est dans la limite gauche de right_box."""
    if left_box is None or right_box is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box

    return x1_1 > x0_2 and x0_1 < x0_2 and (y0_1 <= y0_2 <= y1_1
                                            or y0_1 <= y1_2 <= y1_1)


def _right_intersect(left_box, right_box):
    """Vérifie si les box s'intersectent sur leur limite droite, c'est-à-dire si la limite gauche de left_box est dans la limite droite de right_box."""
    if left_box is None or right_box is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box

    return x0_1 < x1_2 and x1_1 > x1_2 and (y0_1 <= y0_2 <= y1_1
                                            or y0_1 <= y1_2 <= y1_1)


def _is_vertical_full_overlap(box1, box2, x_torlence=2):
    """Direction x : soit box1 contient box2, soit box2 contient box1. Pas de chevauchement partiel. Direction y : box1 et box2 se chevauchent."""
    # Analyse des coordonnées des box
    x11, y11, x12, y12 = box1  # Coordonnées coin supérieur gauche et inférieur droit (x1, y1, x2, y2)
    x21, y21, x22, y22 = box2

    # Sur l'axe x, box1 contient box2 ou box2 contient box1
    contains_in_x = (x11 - x_torlence <= x21 and x12 + x_torlence >= x22) or (
        x21 - x_torlence <= x11 and x22 + x_torlence >= x12)

    # Sur l'axe y, box1 et box2 se chevauchent
    overlap_in_y = not (y12 < y21 or y11 > y22)

    return contains_in_x and overlap_in_y


def _is_bottom_full_overlap(box1, box2, y_tolerance=2):
    """Vérifie si le bas de box1 et le haut de box2 se chevauchent légèrement, avec une tolérance définie par y_tolerance. Différent de _is_vertical_full_overlap car permet un léger chevauchement en x avec une certaine tolérance."""
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    tolerance_margin = 2
    is_xdir_full_overlap = (
        (x0_1 - tolerance_margin <= x0_2 <= x1_1 + tolerance_margin
         and x0_1 - tolerance_margin <= x1_2 <= x1_1 + tolerance_margin)
        or (x0_2 - tolerance_margin <= x0_1 <= x1_2 + tolerance_margin
            and x0_2 - tolerance_margin <= x1_1 <= x1_2 + tolerance_margin))

    return y0_2 < y1_1 and 0 < (y1_1 -
                                y0_2) < y_tolerance and is_xdir_full_overlap


def _is_left_overlap(
    box1,
    box2,
):
    """Vérifie si le côté gauche de box1 chevauche box2. En Y, peut être partiel ou total, indépendamment de la position relative en hauteur des box."""

    def __overlap_y(Ay1, Ay2, By1, By2):
        return max(0, min(Ay2, By2) - max(Ay1, By1))

    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    y_overlap_len = __overlap_y(y0_1, y1_1, y0_2, y1_2)
    ratio_1 = 1.0 * y_overlap_len / (y1_1 - y0_1) if y1_1 - y0_1 != 0 else 0
    ratio_2 = 1.0 * y_overlap_len / (y1_2 - y0_2) if y1_2 - y0_2 != 0 else 0
    vertical_overlap_cond = ratio_1 >= 0.5 or ratio_2 >= 0.5

    return x0_1 <= x0_2 <= x1_1 and vertical_overlap_cond


def __is_overlaps_y_exceeds_threshold(bbox1,
                                      bbox2,
                                      overlap_ratio_threshold=0.8):
    """Vérifie si deux bbox se chevauchent sur l'axe y et si ce chevauchement dépasse 80% de la hauteur de la plus petite bbox"""
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold


def calculate_iou(bbox1, bbox2):
    """Calcule l'Intersection sur Union (IOU) de deux boîtes englobantes.

    Args:
        bbox1 (list[float]): Coordonnées de la première boîte, format [x1, y1, x2, y2] où (x1, y1) est le coin supérieur gauche et (x2, y2) le coin inférieur droit.
        bbox2 (list[float]): Coordonnées de la deuxième boîte, même format que bbox1.

    Returns:
        float: L'IOU des deux boîtes, valeur entre [0, 1].
    """
    # Déterminer les coordonnées du rectangle d'intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Surface de la zone de chevauchement
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Surface des deux rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calcul de l'IOU en divisant l'intersection par la somme des aires moins l'intersection
    iou = intersection_area / float(bbox1_area + bbox2_area -
                                    intersection_area)
    return iou


def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    """Calcule le ratio entre la surface de chevauchement et la surface de la plus petite boîte."""
    # Déterminer les coordonnées du rectangle d'intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Surface de la zone de chevauchement
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]),
                        (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """Calcule le ratio entre la surface de chevauchement et la surface de bbox1."""
    # Déterminer les coordonnées du rectangle d'intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Surface de la zone de chevauchement
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    """Utilise calculate_overlap_area_2_minbox_area_ratio pour calculer le ratio de chevauchement par rapport à la plus petite boîte.
    Si le ratio est supérieur à la valeur donnée, retourne la plus petite bbox, sinon retourne None."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    if overlap_ratio > ratio:
        if area1 <= area2:
            return bbox1
        else:
            return bbox2
    else:
        return None


def get_bbox_in_boundary(bboxes: list, boundary: tuple) -> list:
    x0, y0, x1, y1 = boundary
    new_boxes = [
        box for box in bboxes
        if box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1
    ]
    return new_boxes


def is_vbox_on_side(bbox, width, height, side_threshold=0.2):
    """Détermine si une bbox est sur le bord de la page PDF."""
    x0, x1 = bbox[0], bbox[2]
    if x1 <= width * side_threshold or x0 >= width * (1 - side_threshold):
        return True
    return False


def find_top_nearest_text_bbox(pymu_blocks, obj_bbox):
    tolerance_margin = 4
    top_boxes = [
        box for box in pymu_blocks
        if obj_bbox[1] - box['bbox'][3] >= -tolerance_margin
        and not _is_in(box['bbox'], obj_bbox)
    ]
    # Puis trouve ceux qui se chevauchent en X
    top_boxes = [
        box for box in top_boxes if any([
            obj_bbox[0] - tolerance_margin <= box['bbox'][0] <= obj_bbox[2] +
            tolerance_margin, obj_bbox[0] -
            tolerance_margin <= box['bbox'][2] <= obj_bbox[2] +
            tolerance_margin, box['bbox'][0] -
            tolerance_margin <= obj_bbox[0] <= box['bbox'][2] +
            tolerance_margin, box['bbox'][0] -
            tolerance_margin <= obj_bbox[2] <= box['bbox'][2] +
            tolerance_margin
        ])
    ]

    # Puis trouve celui avec le plus grand y1
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x['bbox'][3], reverse=True)
        return top_boxes[0]
    else:
        return None


def find_bottom_nearest_text_bbox(pymu_blocks, obj_bbox):
    bottom_boxes = [
        box for box in pymu_blocks if box['bbox'][1] -
        obj_bbox[3] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # Puis trouve ceux qui se chevauchent en X
    bottom_boxes = [
        box for box in bottom_boxes if any([
            obj_bbox[0] - 2 <= box['bbox'][0] <= obj_bbox[2] + 2, obj_bbox[0] -
            2 <= box['bbox'][2] <= obj_bbox[2] + 2, box['bbox'][0] -
            2 <= obj_bbox[0] <= box['bbox'][2] + 2, box['bbox'][0] -
            2 <= obj_bbox[2] <= box['bbox'][2] + 2
        ])
    ]

    # Puis trouve celui avec le plus petit y0
    if len(bottom_boxes) > 0:
        bottom_boxes.sort(key=lambda x: x['bbox'][1], reverse=False)
        return bottom_boxes[0]
    else:
        return None


def find_left_nearest_text_bbox(pymu_blocks, obj_bbox):
    """Trouve le bloc de texte le plus proche à gauche."""
    left_boxes = [
        box for box in pymu_blocks if obj_bbox[0] -
        box['bbox'][2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # Puis trouve ceux qui se chevauchent en X
    left_boxes = [
        box for box in left_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # Puis trouve celui avec le plus grand x1
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x['bbox'][2], reverse=True)
        return left_boxes[0]
    else:
        return None


def find_right_nearest_text_bbox(pymu_blocks, obj_bbox):
    """Trouve le bloc de texte le plus proche à droite."""
    right_boxes = [
        box for box in pymu_blocks if box['bbox'][0] -
        obj_bbox[2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # Puis trouve ceux qui se chevauchent en X
    right_boxes = [
        box for box in right_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # Puis trouve celui avec le plus petit x0
    if len(right_boxes) > 0:
        right_boxes.sort(key=lambda x: x['bbox'][0], reverse=False)
        return right_boxes[0]
    else:
        return None


def bbox_relative_pos(bbox1, bbox2):
    """Détermine la position relative de deux rectangles.

    Args:
        bbox1: Un tuple de 4 éléments représentant les coordonnées du premier rectangle (x1, y1, x1b, y1b)
        bbox2: Un tuple de 4 éléments représentant les coordonnées du second rectangle (x2, y2, x2b, y2b)

    Returns:
        Un tuple de 4 booléens (left, right, bottom, top) indiquant la position relative de bbox1 par rapport à bbox2
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1, bbox2):
    """Calcule la distance entre deux rectangles.

    Args:
        bbox1 (tuple): Coordonnées du premier rectangle (x1, y1, x2, y2)
        bbox2 (tuple): Coordonnées du second rectangle (x1, y1, x2, y2)

    Returns:
        float: Distance entre les rectangles.
    """

    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 +
                         (point1[1] - point2[1])**2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_overlap_area(bbox1, bbox2):
    """Calcule la surface de chevauchement entre box1 et box2."""
    # Déterminer les coordonnées du rectangle d'intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Surface de la zone de chevauchement
    return (x_right - x_left) * (y_bottom - y_top)


def calculate_vertical_projection_overlap_ratio(block1, block2):
    """
    Calcule la proportion de l'axe x couverte par la projection verticale de deux blocs.

    Args:
        block1 (tuple): Coordonnées du premier bloc (x0, y0, x1, y1)
        block2 (tuple): Coordonnées du second bloc (x0, y0, x1, y1)

    Returns:
        float: Proportion de l'axe x couverte par la projection verticale des deux blocs
    """
    x0_1, _, x1_1, _ = block1
    x0_2, _, x1_2, _ = block2

    # Calcul de l'intersection des coordonnées x
    x_left = max(x0_1, x0_2)
    x_right = min(x1_1, x1_2)

    if x_right < x_left:
        return 0.0

    # Longueur de l'intersection
    intersection_length = x_right - x_left

    # Longueur de la projection sur l'axe x du premier bloc
    block1_length = x1_1 - x0_1

    if block1_length == 0:
        return 0.0

    # Proportion de l'axe x couverte par l'intersection
    return intersection_length / block1_length
