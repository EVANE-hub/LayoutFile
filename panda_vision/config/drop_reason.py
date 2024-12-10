class DropReason:
    TEXT_BLCOK_HOR_OVERLAP = 'text_block_horizontal_overlap'  # blocs de texte se chevauchant horizontalement, empêchant la localisation précise de l'ordre du texte
    USEFUL_BLOCK_HOR_OVERLAP = (
        'useful_block_horizontal_overlap'  # chevauchement horizontal des blocs à conserver
    )
    COMPLICATED_LAYOUT = 'complicated_layout'  # mise en page complexe, non prise en charge pour le moment
    TOO_MANY_LAYOUT_COLUMNS = 'too_many_layout_columns'  # ne prend pas en charge plus de 2 colonnes actuellement
    COLOR_BACKGROUND_TEXT_BOX = 'color_background_text_box'  # PDF avec des blocs colorés qui modifient l'ordre de lecture, les blocs de texte avec fond coloré ne sont pas pris en charge
    HIGH_COMPUTATIONAL_lOAD_BY_IMGS = (
        'high_computational_load_by_imgs'  # contient des images spéciales, charge de calcul trop élevée, donc abandonné
    )
    HIGH_COMPUTATIONAL_lOAD_BY_SVGS = (
        'high_computational_load_by_svgs'  # SVG spéciaux, charge de calcul trop élevée, donc abandonné
    )
    HIGH_COMPUTATIONAL_lOAD_BY_TOTAL_PAGES = 'high_computational_load_by_total_pages'  # charge de calcul excessive, consommation trop élevée avec la méthode actuelle
    MISS_DOC_LAYOUT_RESULT = 'missing doc_layout_result'  # échec de l'analyse de la mise en page
    Exception = '_exception'  # exception survenue pendant l'analyse
    ENCRYPTED = 'encrypted'  # PDF est crypté
    EMPTY_PDF = 'total_page=0'  # nombre total de pages PDF est 0
    NOT_IS_TEXT_PDF = 'not_is_text_pdf'  # pas un PDF texte, impossible à analyser directement
    DENSE_SINGLE_LINE_BLOCK = 'dense_single_line_block'  # impossible de distinguer clairement les paragraphes
    TITLE_DETECTION_FAILED = 'title_detection_failed'  # échec de la détection des titres
    TITLE_LEVEL_FAILED = (
        'title_level_failed'  # échec de l'analyse des niveaux de titre (ex: titre niveau 1, 2, 3)
    )
    PARA_SPLIT_FAILED = 'para_split_failed'  # échec de la reconnaissance des paragraphes
    PARA_MERGE_FAILED = 'para_merge_failed'  # échec de la fusion des paragraphes
    NOT_ALLOW_LANGUAGE = 'not_allow_language'  # langue non prise en charge
    SPECIAL_PDF = 'special_pdf'
    PSEUDO_SINGLE_COLUMN = 'pseudo_single_column'  # impossible de déterminer précisément les colonnes de texte
    CAN_NOT_DETECT_PAGE_LAYOUT = 'can_not_detect_page_layout'  # impossible d'analyser la mise en page
    NEGATIVE_BBOX_AREA = 'negative_bbox_area'  # mise à l'échelle entraînant une surface de bbox négative
    OVERLAP_BLOCKS_CAN_NOT_SEPARATION = (
        'overlap_blocks_can_t_separation'  # impossible de séparer les blocs qui se chevauchent
    )
