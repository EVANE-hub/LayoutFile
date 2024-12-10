"""Entrée: chemin S3, un par ligne. Sortie: métadonnées du fichier PDF, y compris la longueur, la largeur et la hauteur de toutes les images sur chaque page, ainsi que leur position bbox."""

from collections import Counter

import fitz
from loguru import logger

from panda_vision.config.drop_reason import DropReason
from panda_vision.libs.commons import get_top_percent_list, mymax
from panda_vision.libs.language import detect_lang
from panda_vision.libs.pdf_check import detect_invalid_chars_by_pymupdf

scan_max_page = 50
junk_limit_min = 10


def calculate_max_image_area_per_page(result: list, page_width_pts, page_height_pts):
    max_image_area_per_page = [
        mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz])
        for page_img_sz in result
    ]
    page_area = int(page_width_pts) * int(page_height_pts)
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.6]
    return max_image_area_per_page


def process_image(page, junk_img_bojids=[]):
    page_result = []  # Stocke les informations des quadruplets d'images pour chaque page
    items = page.get_images()
    dedup = set()
    for img in items:
        # Retourne la taille réelle de l'image affichée sur la page. Retourne un tableau, chaque élément est
        img_bojid = img[
            0
        ]  # Est unique globalement dans le fichier PDF, si cette image apparaît plusieurs fois dans le PDF, elle peut être une information indésirable comme un filigrane, en-tête/pied de page etc.
        if img_bojid in junk_img_bojids:  # Si c'est une image indésirable, on la saute
            continue
        recs = page.get_image_rects(img, transform=True)
        if recs:
            rec = recs[0][0]
            x0, y0, x1, y1 = map(int, rec)
            width = x1 - x0
            height = y1 - y0
            if (
                x0,
                y0,
                x1,
                y1,
                img_bojid,
            ) in dedup:  # Il y a des bbox en double ici, pas besoin de les répéter, il faut les supprimer
                continue
            if not all(
                [width, height]
            ):  # Ni la longueur ni la largeur ne peuvent être 0, sinon l'image n'est pas visible et n'a pas de sens
                continue
            dedup.add((x0, y0, x1, y1, img_bojid))
            page_result.append([x0, y0, x1, y1, img_bojid])
    return page_result


def get_image_info(doc: fitz.Document, page_width_pts, page_height_pts) -> list:
    """Retourne les quadruplets d'images pour chaque page, plusieurs images par page.

    :param doc:
    :return:
    """
    # Utilise Counter pour compter les occurrences de img_bojid
    img_bojid_counter = Counter(img[0] for page in doc for img in page.get_images())
    # Trouve les img_bojid qui apparaissent plus de la moitié du nombre de pages

    junk_limit = max(len(doc) * 0.5, junk_limit_min)  # Exempte les documents avec peu de pages

    junk_img_bojids = [
        img_bojid
        for img_bojid, count in img_bojid_counter.items()
        if count >= junk_limit
    ]

    # TODO: Ajouter une vérification, utiliser seulement les 10 premières pages, ces images indésirables doivent remplir deux conditions:
    # non seulement leur fréquence d'apparition doit être suffisamment élevée, mais aussi leur taille relative à la page doit être suffisamment grande
    # et les images doivent avoir des tailles similaires
    # Il y a deux types de versions numérisées et une version texte, il peut y avoir des faux positifs ici
    # Version numérisée 1: chaque page a toutes les images numérisées, caractérisée par une grande proportion d'image, 1 image par page
    # Version numérisée 2: le nombre d'images numérisées stockées augmente par page, caractérisée par une grande proportion d'image,
    # 1 image par page, nécessite de vider junklist et d'analyser les 50 premières pages pour la classification
    # Version texte 1: stocke toutes les images sur chaque page, caractérisée par une faible proportion d'images par page,
    # peut avoir 0 ou plusieurs images par page. Ce type de PDF nécessite un échantillonnage des 10 premières pages pour détecter
    # la taille et le nombre d'images, si conforme il faut vider junklist
    imgs_len_list = [len(page.get_images()) for page in doc]

    special_limit_pages = 10

    # Utilise uniformément les 10 premières pages pour le jugement
    result = []
    break_loop = False
    for i, page in enumerate(doc):
        if break_loop:
            break
        if i >= special_limit_pages:
            break
        page_result = process_image(
            page
        )  # Ne passe pas junk_img_bojids ici, prend toutes les informations d'image des 10 premières pages pour analyse ultérieure
        result.append(page_result)
        for item in result:
            if not any(
                item
            ):  # Si une page n'a pas d'image, c'est une version texte, il faut vérifier si c'est une version texte spéciale
                if (
                    max(imgs_len_list) == min(imgs_len_list)
                    and max(imgs_len_list) >= junk_limit_min
                ):  # Si c'est une version texte spéciale, vide junklist et break
                    junk_img_bojids = []
                else:  # Pas une version texte spéciale, une version texte normale avec des images indésirables, ne vide pas junklist
                    pass
                break_loop = True
                break
    if not break_loop:
        # Obtient les 80% premiers éléments
        top_eighty_percent = get_top_percent_list(imgs_len_list, 0.8)
        # Vérifie si les 80% premiers éléments sont égaux
        if len(set(top_eighty_percent)) == 1 and max(imgs_len_list) >= junk_limit_min:
            # Si les 10 premières pages ont toutes des images et le même nombre, vérifie la proportion de la taille des images
            # par rapport à la page pour décider s'il faut vider junklist
            max_image_area_per_page = calculate_max_image_area_per_page(
                result, page_width_pts, page_height_pts
            )
            if (
                len(max_image_area_per_page) < 0.8 * special_limit_pages
            ):  # Les 10 premières pages ne sont pas toutes de grandes images, peut être un PDF texte, vide la liste d'images indésirables
                junk_img_bojids = []
            else:  # Les 10 premières pages ont toutes des images, 80% sont de grandes images, même nombre d'images par page et nombreuses,
                # c'est une version numérisée 1, pas besoin de vider junklist
                pass
        else:  # Nombre d'images différent par page, doit vider junklist et analyser les 50 premières pages
            junk_img_bojids = []

    # Entre dans le processus formel d'obtention des informations d'image des 50 premières pages
    result = []
    for i, page in enumerate(doc):
        if i >= scan_max_page:
            break
        page_result = process_image(page, junk_img_bojids)
        # logger.info(f"page {i} img_len: {len(page_result)}")
        result.append(page_result)

    return result, junk_img_bojids


def get_pdf_page_size_pts(doc: fitz.Document):
    page_cnt = len(doc)
    l: int = min(page_cnt, 50)
    # Met toutes les largeurs et hauteurs dans deux listes et prend la médiane pour chacune
    # (a rencontré un PDF avec des pages horizontales dans un document vertical, causant une inversion largeur/hauteur)
    page_width_list = []
    page_height_list = []
    for i in range(l):
        page = doc[i]
        page_rect = page.rect
        page_width_list.append(page_rect.width)
        page_height_list.append(page_rect.height)

    page_width_list.sort()
    page_height_list.sort()

    median_width = page_width_list[len(page_width_list) // 2]
    median_height = page_height_list[len(page_height_list) // 2]

    return median_width, median_height


def get_pdf_textlen_per_page(doc: fitz.Document):
    text_len_lst = []
    for page in doc:
        # Obtient tous les blocs contenant images et texte
        # text_block = page.get_text("blocks")
        # Obtient tous les blocs de texte
        # text_block = page.get_text("words")
        # text_block_len = sum([len(t[4]) for t in text_block])
        # Obtient tout le texte en str
        text_block = page.get_text('text')
        text_block_len = len(text_block)
        # logger.info(f"page {page.number} text_block_len: {text_block_len}")
        text_len_lst.append(text_block_len)

    return text_len_lst


def get_pdf_text_layout_per_page(doc: fitz.Document):
    """Détermine si la mise en page du texte est horizontale, verticale ou inconnue pour chaque page du document PDF.

    Args:
        doc (fitz.Document): Objet document PDF.

    Returns:
        List[str]: Mise en page du texte pour chaque page (horizontal, vertical, inconnu).
    """
    text_layout_list = []

    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break
        # Crée des compteurs pour les lignes verticales et horizontales de chaque page
        vertical_count = 0
        horizontal_count = 0
        text_dict = page.get_text('dict')
        if 'blocks' in text_dict:
            for block in text_dict['blocks']:
                if 'lines' in block:
                    for line in block['lines']:
                        # Obtient les coordonnées des sommets de la bbox
                        x0, y0, x1, y1 = line['bbox']
                        # Calcule largeur et hauteur de la bbox
                        width = x1 - x0
                        height = y1 - y0
                        # Calcule l'aire de la bbox
                        area = width * height
                        font_sizes = []
                        for span in line['spans']:
                            if 'size' in span:
                                font_sizes.append(span['size'])
                        if len(font_sizes) > 0:
                            average_font_size = sum(font_sizes) / len(font_sizes)
                        else:
                            average_font_size = (
                                10  # Certaines lignes n'ont pas de font_size, fixe un seuil de 100
                            )
                        if (
                            area <= average_font_size**2
                        ):  # Vérifie si l'aire de la bbox est inférieure au carré de la taille moyenne de police,
                            # impossible de calculer l'orientation pour un seul caractère
                            continue
                        else:
                            if 'wmode' in line:  # Détermine l'orientation du texte par wmode
                                if line['wmode'] == 1:  # Vérifie si c'est du texte vertical
                                    vertical_count += 1
                                elif line['wmode'] == 0:  # Vérifie si c'est du texte horizontal
                                    horizontal_count += 1
        # print(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")
        # Détermine la mise en page de chaque page
        if vertical_count == 0 and horizontal_count == 0:  # Page sans texte, impossible de déterminer
            text_layout_list.append('unknow')
            continue
        else:
            if vertical_count > horizontal_count:  # Plus de lignes verticales qu'horizontales sur la page
                text_layout_list.append('vertical')
            else:  # Plus de lignes horizontales que verticales sur la page
                text_layout_list.append('horizontal')
        # logger.info(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")
    return text_layout_list


"""Définit une exception personnalisée pour les PDF avec trop de SVG par page"""


class PageSvgsTooManyError(Exception):
    def __init__(self, message='Page SVGs are too many'):
        self.message = message
        super().__init__(self.message)


def get_svgs_per_page(doc: fitz.Document):
    svgs_len_list = []
    for page_id, page in enumerate(doc):
        # svgs = page.get_drawings()
        svgs = page.get_cdrawings()  # Passe à get_cdrawings, plus efficace
        len_svgs = len(svgs)
        if len_svgs >= 3000:
            raise PageSvgsTooManyError()
        else:
            svgs_len_list.append(len_svgs)
        # logger.info(f"page_id: {page_id}, svgs_len: {len(svgs)}")
    return svgs_len_list


def get_imgs_per_page(doc: fitz.Document):
    imgs_len_list = []
    for page_id, page in enumerate(doc):
        imgs = page.get_images()
        imgs_len_list.append(len(imgs))
        # logger.info(f"page_id: {page}, imgs_len: {len(imgs)}")

    return imgs_len_list


def get_language(doc: fitz.Document):
    """
    Obtient la langue du document PDF.
    Args:
        doc (fitz.Document): Objet document PDF.
    Returns:
        str: Langue du document, ex: "en-US".
    """
    language_lst = []
    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break
        # Obtient tout le texte en str
        text_block = page.get_text('text')
        page_language = detect_lang(text_block)
        language_lst.append(page_language)

        # logger.info(f"page_id: {page_id}, page_language: {page_language}")

    # Compte le nombre d'occurrences de chaque langue
    count_dict = Counter(language_lst)
    # Retourne la langue la plus fréquente
    language = max(count_dict, key=count_dict.get)
    return language


def check_invalid_chars(pdf_bytes):
    """Détection des caractères invalides."""
    return detect_invalid_chars_by_pymupdf(pdf_bytes)


def pdf_meta_scan(pdf_bytes: bytes):
    """
    :param s3_pdf_path:
    :param pdf_bytes: données binaires du fichier PDF
    Plusieurs dimensions d'évaluation: chiffrement, protection par mot de passe, taille du papier, nombre total de pages, possibilité d'extraction du texte
    """
    doc = fitz.open('pdf', pdf_bytes)
    is_needs_password = doc.needs_pass
    is_encrypted = doc.is_encrypted
    total_page = len(doc)
    if total_page == 0:
        logger.warning(f'drop this pdf, drop_reason: {DropReason.EMPTY_PDF}')
        result = {'_need_drop': True, '_drop_reason': DropReason.EMPTY_PDF}
        return result
    else:
        page_width_pts, page_height_pts = get_pdf_page_size_pts(doc)
        # logger.info(f"page_width_pts: {page_width_pts}, page_height_pts: {page_height_pts}")

        # svgs_per_page = get_svgs_per_page(doc)
        # logger.info(f"svgs_per_page: {svgs_per_page}")
        imgs_per_page = get_imgs_per_page(doc)
        # logger.info(f"imgs_per_page: {imgs_per_page}")

        image_info_per_page, junk_img_bojids = get_image_info(
            doc, page_width_pts, page_height_pts
        )
        # logger.info(f"image_info_per_page: {image_info_per_page}, junk_img_bojids: {junk_img_bojids}")
        text_len_per_page = get_pdf_textlen_per_page(doc)
        # logger.info(f"text_len_per_page: {text_len_per_page}")
        text_layout_per_page = get_pdf_text_layout_per_page(doc)
        # logger.info(f"text_layout_per_page: {text_layout_per_page}")
        text_language = get_language(doc)
        # logger.info(f"text_language: {text_language}")
        invalid_chars = check_invalid_chars(pdf_bytes)
        # logger.info(f"invalid_chars: {invalid_chars}")

        # Sortie finale en JSON
        res = {
            'is_needs_password': is_needs_password,
            'is_encrypted': is_encrypted,
            'total_page': total_page,
            'page_width_pts': int(page_width_pts),
            'page_height_pts': int(page_height_pts),
            'image_info_per_page': image_info_per_page,
            'text_len_per_page': text_len_per_page,
            'text_layout_per_page': text_layout_per_page,
            'text_language': text_language,
            # "svgs_per_page": svgs_per_page,
            'imgs_per_page': imgs_per_page,  # Ajoute la liste du nombre d'images par page
            'junk_img_bojids': junk_img_bojids,  # Ajoute la liste des bojid d'images indésirables
            'invalid_chars': invalid_chars,
            'metadata': doc.metadata,
        }
        # logger.info(json.dumps(res, ensure_ascii=False))
        return res


if __name__ == '__main__':
    pass
    # "D:\project/20231108code-clean\pdf_cost_time\竖排例子\净空法师-大乘无量寿.pdf"
    # "D:\project/20231108code-clean\pdf_cost_time\竖排例子\三国演义_繁体竖排版.pdf"
    # "D:\project/20231108code-clean\pdf_cost_time\scihub\scihub_86800000\libgen.scimag86880000-86880999.zip_10.1021/acsami.1c03109.s002.pdf"
    # "D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_18600000/libgen.scimag18645000-18645999.zip_10.1021/om3006239.pdf"
    # file_content = read_file("D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_31000000/libgen.scimag31098000-31098999.zip_10.1109/isit.2006.261791.pdf","")  # noqa: E501
    # file_content = read_file("D:\project/20231108code-clean\pdf_cost_time\竖排例子\净空法师_大乘无量寿.pdf","")
    # doc = fitz.open("pdf", file_content)
    # text_layout_lst = get_pdf_text_layout_per_page(doc)
    # print(text_layout_lst)
