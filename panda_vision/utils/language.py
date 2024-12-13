import os
import unicodedata
from fast_langdetect import detect_language

if not os.getenv("FTLANG_CACHE"):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(current_dir)
    ftlang_cache_dir = os.path.join(root_dir, 'resources', 'fasttext-langdetect')
    os.environ["FTLANG_CACHE"] = str(ftlang_cache_dir)


def detect_lang(text: str) -> str:
    """
    Détecter la langue du texte donné.
    
    :param text: Le texte dont on veut détecter la langue.
    :return: Le code de la langue détectée en minuscules, ou une chaîne vide si la détection échoue.
    """
    if len(text) == 0:
        return ""
    try:
        lang_upper = detect_language(text)
    except:
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        lang_upper = detect_language(html_no_ctrl_chars)
    try:
        lang = lang_upper.lower()
    except:
        lang = ""
    return lang