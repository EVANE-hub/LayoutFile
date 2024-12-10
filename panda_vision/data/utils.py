
import fitz
import numpy as np

from panda_vision.utils.annotations import ImportPIL


@ImportPIL
def fitz_doc_to_image(doc, dpi=200) -> dict:
    """Convertit fitz.Document en image, puis convertit l'image en tableau numpy.

    Args:
        doc (_type_): page pymudoc
        dpi (int, optional): réinitialise le dpi. Par défaut 200.

    Returns:
        dict:  {'img': tableau numpy, 'width': largeur, 'height': hauteur }
    """
    from PIL import Image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    # Si la largeur ou la hauteur dépasse 4500 après mise à l'échelle, ne pas redimensionner davantage.
    if pm.width > 4500 or pm.height > 4500:
        pm = doc.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)

    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}

    return img_dict
