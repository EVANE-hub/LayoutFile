class DropReason:
    TEXT_BLOCK_HOR_OVERLAP = 'text_block_horizontal_overlap'  # Blocs de texte se chevauchant horizontalement, empêchant la localisation précise de l'ordre du texte
    USEFUL_BLOCK_HOR_OVERLAP = 'useful_block_horizontal_overlap'  # Chevauchement horizontal des blocs à conserver
    COMPLICATED_LAYOUT = 'complicated_layout'  # Mise en page complexe, non prise en charge pour le moment
    TOO_MANY_LAYOUT_COLUMNS = 'too_many_layout_columns'  # Ne prend pas en charge plus de 2 colonnes actuellement
    COLOR_BACKGROUND_TEXT_BOX = 'color_background_text_box'  # Les blocs de texte avec fond coloré ne sont pas pris en charge
    HIGH_COMPUTATIONAL_LOAD_BY_IMAGES = 'high_computational_load_by_images'  # Contient des images spéciales, charge de calcul trop élevée
    HIGH_COMPUTATIONAL_LOAD_BY_SVGS = 'high_computational_load_by_svgs'  # Contient des SVG spéciaux, charge de calcul trop élevée
    HIGH_COMPUTATIONAL_LOAD_BY_TOTAL_PAGES = 'high_computational_load_by_total_pages'  # Charge de calcul excessive due au nombre total de pages
    MISSING_DOC_LAYOUT_RESULT = 'missing_doc_layout_result'  # Échec de l'analyse de la mise en page
    EXCEPTION = '_exception'  # Exception survenue pendant l'analyse
    ENCRYPTED = 'encrypted'  # Le PDF est crypté
    EMPTY_PDF = 'total_page=0'  # Le PDF ne contient aucune page
    NOT_A_TEXT_PDF = 'not_a_text_pdf'  # Le PDF n'est pas un document texte, analyse impossible
    DENSE_SINGLE_LINE_BLOCK = 'dense_single_line_block'  # Impossible de distinguer clairement les paragraphes
    TITLE_DETECTION_FAILED = 'title_detection_failed'  # Échec de la détection des titres
    TITLE_LEVEL_FAILED = 'title_level_failed'  # Échec de l'analyse des niveaux de titres
    PARA_SPLIT_FAILED = 'para_split_failed'  # Échec de la reconnaissance des paragraphes
    PARA_MERGE_FAILED = 'para_merge_failed'  # Échec de la fusion des paragraphes
    NOT_ALLOWED_LANGUAGE = 'not_allowed_language'  # Langue non prise en charge
    SPECIAL_PDF = 'special_pdf'  # PDF avec des caractéristiques spéciales
    PSEUDO_SINGLE_COLUMN = 'pseudo_single_column'  # Impossible de déterminer précisément les colonnes de texte
    CANNOT_DETECT_PAGE_LAYOUT = 'cannot_detect_page_layout'  # Impossible d'analyser la mise en page
    NEGATIVE_BBOX_AREA = 'negative_bbox_area'  # Surface de boîte englobante négative après mise à l'échelle
    OVERLAP_BLOCKS_CANNOT_SEPARATE = 'overlap_blocks_cannot_separate'  # Impossible de séparer les blocs qui se chevauchent
