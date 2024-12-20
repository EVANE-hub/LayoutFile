"""Retourne le triplet AK, SK, endpoint S3 correspondant au nom du bucket."""

import json
import os

from loguru import logger

from panda_vision.config.constants import MODEL_NAME
from panda_vision.libs.commons import parse_bucket_key

# Définition de la constante du nom du fichier de configuration
CONFIG_FILE_NAME = os.getenv('PANDA_VISION_CONFIG_JSON', 'PANDA-VISION-CONFIG.json')


def read_config():
    if os.path.isabs(CONFIG_FILE_NAME):
        config_file = CONFIG_FILE_NAME
    else:
        home_dir = os.path.expanduser('~')
        config_file = os.path.join(home_dir, CONFIG_FILE_NAME)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} introuvable')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_s3_config(bucket_name: str):
    """Lecture depuis ~/PANDA-VISION-CONFIG.json."""
    config = read_config()

    bucket_info = config.get('bucket_info')
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info['[default]']
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f'ak, sk ou endpoint non trouvé dans {CONFIG_FILE_NAME}')

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def get_local_models_dir():
    config = read_config()
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(f"'models-dir' non trouvé dans {CONFIG_FILE_NAME}, utilisation de '/tmp/models' par défaut")
        return '/tmp/models'
    else:
        return models_dir


def get_local_layoutreader_model_dir():
    config = read_config()
    layoutreader_model_dir = config.get('layoutreader-model-dir')
    if layoutreader_model_dir is None or not os.path.exists(layoutreader_model_dir):
        home_dir = os.path.expanduser('~')
        layoutreader_at_modelscope_dir_path = os.path.join(home_dir, '.cache/modelscope/hub/ppaanngggg/layoutreader')
        logger.warning(f"'layoutreader-model-dir' n'existe pas, utilisation de {layoutreader_at_modelscope_dir_path} par défaut")
        return layoutreader_at_modelscope_dir_path
    else:
        return layoutreader_model_dir


def get_device():
    config = read_config()
    device = config.get('device-mode')
    if device is None:
        logger.warning(f"'device-mode' non trouvé dans {CONFIG_FILE_NAME}, utilisation de 'cpu' par défaut")
        return 'cpu'
    else:
        return device


def get_table_recog_config():
    config = read_config()
    table_config = config.get('table-config')
    if table_config is None:
        logger.warning(f"'table-config' non trouvé dans {CONFIG_FILE_NAME}, utilisation de 'False' par défaut")
        return json.loads(f'{{"model": "{MODEL_NAME.RAPID_TABLE}","enable": false, "max_time": 400}}')
    else:
        return table_config


def get_layout_config():
    config = read_config()
    layout_config = config.get('layout-config')
    if layout_config is None:
        logger.warning(f"'layout-config' non trouvé dans {CONFIG_FILE_NAME}, utilisation de '{MODEL_NAME.LAYOUTLMv3}' par défaut")
        return json.loads(f'{{"model": "{MODEL_NAME.LAYOUTLMv3}"}}')
    else:
        return layout_config


def get_formula_config():
    config = read_config()
    formula_config = config.get('formula-config')
    if formula_config is None:
        logger.warning(f"'formula-config' non trouvé dans {CONFIG_FILE_NAME}, utilisation de 'True' par défaut")
        return json.loads(f'{{"mfd_model": "{MODEL_NAME.YOLO_V8_MFD}","mfr_model": "{MODEL_NAME.UniMerNet_v2_Small}","enable": true}}')
    else:
        return formula_config


if __name__ == '__main__':
    ak, sk, endpoint = get_s3_config('llm-raw')
