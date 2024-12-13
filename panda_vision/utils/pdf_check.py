import fitz
import numpy as np
from loguru import logger

def calculate_sample_count(total_page: int) -> int:
    """
    Calcule le nombre de pages à échantillonner en fonction du nombre total de pages.
    """
    return min(10, total_page)

def extract_pages(src_pdf_bytes: bytes) -> fitz.Document:
    pdf_docs = fitz.open("pdf", src_pdf_bytes)
    total_page = len(pdf_docs)
    if total_page == 0:
        # Si le PDF n'a pas de pages, retourne un document vide
        logger.warning("PDF is empty, return empty document")
        return fitz.Document()

    select_page_cnt = calculate_sample_count(total_page)
    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    sample_docs = fitz.Document()

    try:
        for index in page_num:
            sample_docs.insert_pdf(pdf_docs, from_page=int(index), to_page=int(index))
    except Exception as e:
        logger.exception(e)

    return sample_docs

def count_replacement_characters(text: str) -> int:
    """
    Compte le nombre de caractères 0xfffd dans la chaîne.
    """
    return text.count('\ufffd')

def detect_invalid_chars_by_pymupdf(src_pdf_bytes: bytes) -> bool:
    sample_docs = extract_pages(src_pdf_bytes)
    doc_text = "".join(page.get_text('text', flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP) for page in sample_docs)
    text_len = len(doc_text)
    uffd_count = count_replacement_characters(doc_text)
    uffd_chars_radio = uffd_count / text_len if text_len > 0 else 0

    logger.info(f"uffd_count: {uffd_count}, text_len: {text_len}, uffd_chars_radio: {uffd_chars_radio}")
    '''Quand plus de 1% du texte est illisible, on considère le document comme corrompu'''
    return uffd_chars_radio <= 0.01
