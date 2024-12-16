"""Champs personnalisés au niveau span."""
# Indique si le span est fusionné sur plusieurs pages
CROSS_PAGE = 'cross_page'

"""
Champs personnalisés au niveau block
"""
# Indique si les lignes dans le block ont été supprimées
LINES_DELETED = 'lines_deleted'

# Valeur par défaut du temps maximum de reconnaissance de tableau
TABLE_MAX_TIME_VALUE = 400

# Longueur maximale du résultat de tableau pp
TABLE_MAX_LEN = 480

# Dictionnaire de structure de tableau maître
TABLE_MASTER_DICT = 'table_master_structure_dict.txt'

# Répertoire maître de tableau
TABLE_MASTER_DIR = 'table_structure_tablemaster_infer/'

# Répertoire du modèle de détection pp
DETECT_MODEL_DIR = 'ch_PP-OCRv4_det_infer'

# Répertoire du modèle de reconnaissance pp
REC_MODEL_DIR = 'ch_PP-OCRv4_rec_infer'

class MODEL_NAME:
    # Algorithme de structure de tableau pp
    TABLE_MASTER = 'tablemaster'
    STRUCT_EQTABLE = 'struct_eqtable'
    DocLayout_YOLO = 'doclayout_yolo'
    LAYOUTLMv3 = 'layoutlmv3'
    YOLO_V8_MFD = 'yolo_v8_mfd'
    UniMerNet_v2_Small = 'unimernet_base'
    RAPID_TABLE = 'rapid_table'
