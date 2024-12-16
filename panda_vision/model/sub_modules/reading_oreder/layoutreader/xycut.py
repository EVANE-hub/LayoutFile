from typing import List, Tuple, Optional
import cv2
import numpy as np


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """Calcule l'histogramme de projection des boîtes englobantes.

    Args:
        boxes: Tableau de forme [N, 4] contenant les coordonnées des boîtes
        axis: 0 pour projection horizontale, 1 pour projection verticale

    Returns:
        Histogramme de projection 1D
    """
    if axis not in (0, 1):
        raise ValueError("L'axe doit être 0 ou 1")
    
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    
    # Vectorisation possible avec np.add.at
    for start, end in boxes[:, axis::2]:
        res[start:end] += 1
    return res


def split_projection_profile(
    arr_values: np.ndarray, 
    min_value: float, 
    min_gap: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Divise le profil de projection en groupes."""
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return None

    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1]) + 1

    return arr_start, arr_end


def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]) -> None:
    """Applique récursivement l'algorithme de découpage XY."""
    if not len(boxes) == len(indices):
        return

    _indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[_indices]
    y_sorted_indices = indices[_indices]

    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)
    if not pos_y:
        return

    arr_y0, arr_y1 = pos_y
    for r0, r1 in zip(arr_y0, arr_y1):
        _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

        y_sorted_boxes_chunk = y_sorted_boxes[_indices]
        y_sorted_indices_chunk = y_sorted_indices[_indices]

        _indices = y_sorted_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
        x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]

        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        pos_x = split_projection_profile(x_projection, 0, 1)
        if not pos_x:
            continue

        arr_x0, arr_x1 = pos_x
        if len(arr_x0) == 1:
            res.extend(x_sorted_indices_chunk)
            continue

        for c0, c1 in zip(arr_x0, arr_x1):
            _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (
                x_sorted_boxes_chunk[:, 0] < c1
            )
            recursive_xy_cut(
                x_sorted_boxes_chunk[_indices], x_sorted_indices_chunk[_indices], res
            )


def points_to_bbox(points: np.ndarray) -> List[float]:
    """Convertit une liste de points en boîte englobante."""
    if len(points) != 8:
        raise ValueError("Le tableau de points doit contenir exactement 8 valeurs")

    x_coords = points[::2]
    y_coords = points[1::2]
    
    return [
        max(min(x_coords), 0),
        max(min(y_coords), 0),
        max(max(x_coords), 0),
        max(max(y_coords), 0)
    ]


def bbox2points(bbox: List[float]) -> List[float]:
    """Convertit une boîte englobante en liste de points."""
    left, top, right, bottom = bbox
    return [left, top, right, top, right, bottom, left, bottom]


def vis_polygon(
    img: np.ndarray,
    points: np.ndarray,
    thickness: int = 2,
    color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """Dessine un polygone sur l'image."""
    points = points.reshape(-1, 2)
    for i in range(4):
        pt1 = tuple(map(int, points[i]))
        pt2 = tuple(map(int, points[(i + 1) % 4]))
        cv2.line(img, pt1, pt2, color=color, thickness=thickness)
    return img


def vis_points(
    img: np.ndarray, points, texts: List[str] = None, color=(0, 200, 0)
) -> np.ndarray:
    """

    Args:
        img:
        points: [N, 8]  8: x1,y1,x2,y2,x3,y3,x3,y4
        texts:
        color:

    Returns:

    """
    points = np.array(points)
    if texts is not None:
        assert len(texts) == points.shape[0]

    for i, _points in enumerate(points):
        vis_polygon(img, _points.reshape(-1, 2), thickness=2, color=color)
        bbox = points_to_bbox(_points)
        left, top, right, bottom = bbox
        cx = (left + right) // 2
        cy = (top + bottom) // 2

        txt = texts[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]

        img = cv2.rectangle(
            img,
            (cx - 5 * len(txt), cy - cat_size[1] - 5),
            (cx - 5 * len(txt) + cat_size[0], cy - 5),
            color,
            -1,
        )

        img = cv2.putText(
            img,
            txt,
            (cx - 5 * len(txt), cy - 5),
            font,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return img


def vis_polygons_with_index(image, points):
    texts = [str(i) for i in range(len(points))]
    res_img = vis_points(image.copy(), points, texts)
    return res_img