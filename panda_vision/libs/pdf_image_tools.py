from io import BytesIO
import cv2
import fitz
import numpy as np
from PIL import Image
from panda_vision.data.data_reader_writer import DataWriter
from panda_vision.libs.commons import join_path
from panda_vision.libs.hash_utils import compute_sha256


def cut_image(bbox: tuple, page_num: int, page: fitz.Page, return_path, imageWriter: DataWriter):
    """À partir de la page page_num, découpe une image jpg selon les coordonnées bbox et retourne le chemin de l'image. save_path doit supporter à la fois s3 et local,
    l'image est stockée sous save_path, avec comme nom de fichier:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , les nombres dans bbox sont arrondis."""
    # Concaténation du nom de fichier
    filename = f'{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}'

    # L'ancienne version retourne le chemin sans le bucket
    img_path = join_path(return_path, filename) if return_path is not None else None

    # La nouvelle version génère un chemin aplati
    img_hash256_path = f'{compute_sha256(img_path)}.jpg'

    # Conversion des coordonnées en objet fitz.Rect
    rect = fitz.Rect(*bbox)
    # Configuration du facteur de zoom à 3x
    zoom = fitz.Matrix(3, 3)
    # Capture de l'image
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)

    imageWriter.write(img_hash256_path, byte_data)

    return img_hash256_path


def cut_image_to_pil_image(bbox: tuple, page: fitz.Page, mode="pillow"):

    # Conversion des coordonnées en objet fitz.Rect
    rect = fitz.Rect(*bbox)
    # Configuration du facteur de zoom à 3x
    zoom = fitz.Matrix(3, 3)
    # Capture de l'image
    pix = page.get_pixmap(clip=rect, matrix=zoom)

    # Conversion des données binaires en objet fichier
    image_file = BytesIO(pix.tobytes(output='png'))
    # Ouverture de l'image avec Pillow
    pil_image = Image.open(image_file)
    if mode == "cv2":
        image_result = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    elif mode == "pillow":
        image_result = pil_image
    else:
        raise ValueError(f"mode: {mode} is not supported.")

    return image_result