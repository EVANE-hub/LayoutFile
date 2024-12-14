from loguru import logger

from panda_vision.config.constants import MODEL_NAME
from panda_vision.model.model_list import AtomicModel
from panda_vision.model.sub_modules.layout.doclayout_yolo.DocLayoutYOLO import DocLayoutYOLOModel
from panda_vision.model.sub_modules.layout.layoutlmv3.model_init import Layoutlmv3_Predictor
from panda_vision.model.sub_modules.mfd.yolov8.YOLOv8 import YOLOv8MFDModel
from panda_vision.model.sub_modules.mfr.unimernet.Unimernet import UnimernetModel
from panda_vision.model.sub_modules.ocr.paddleocr.ppocr_273_mod import ModifiedPaddleOCR
from panda_vision.model.sub_modules.table.rapidtable.rapid_table import RapidTableModel
from panda_vision.model.sub_modules.table.structeqtable.struct_eqtable import StructTableModel
from panda_vision.model.sub_modules.table.tablemaster.tablemaster_paddle import TableMasterPaddleModel


def table_model_init(table_model_type, model_path, max_time, _device_='cpu'):
    models = {
        MODEL_NAME.STRUCT_EQTABLE: lambda: StructTableModel(model_path, max_new_tokens=2048, max_time=max_time),
        MODEL_NAME.TABLE_MASTER: lambda: TableMasterPaddleModel({'model_dir': model_path, 'device': _device_}),
        MODEL_NAME.RAPID_TABLE: RapidTableModel
    }
    
    if table_model_type not in models:
        logger.error('table model type not allow')
        exit(1)
    
    return models[table_model_type]()


def mfd_model_init(weight, device='cpu'):
    mfd_model = YOLOv8MFDModel(weight, device)
    return mfd_model


def mfr_model_init(weight_dir, cfg_path, device='cpu'):
    mfr_model = UnimernetModel(weight_dir, cfg_path, device)
    return mfr_model


def layout_model_init(weight, config_file, device):
    model = Layoutlmv3_Predictor(weight, config_file, device)
    return model


def doclayout_yolo_model_init(weight, device='cpu'):
    model = DocLayoutYOLOModel(weight, device)
    return model


def ocr_model_init(show_log: bool = False,
                   det_db_box_thresh=0.3,
                   lang=None,
                   use_dilation=True,
                   det_db_unclip_ratio=1.8):
    base_params = {
        'show_log': show_log,
        'det_db_box_thresh': det_db_box_thresh,
        'use_dilation': use_dilation,
        'det_db_unclip_ratio': det_db_unclip_ratio
    }
    
    if lang is not None and lang != '':
        base_params['lang'] = lang
        
    return ModifiedPaddleOCR(**base_params)


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):
        key = (atom_model_name, kwargs.get('layout_model_name'), kwargs.get('lang'))
        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]


def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        if kwargs.get('layout_model_name') == MODEL_NAME.LAYOUTLMv3:
            atom_model = layout_model_init(
                kwargs.get('layout_weights'),
                kwargs.get('layout_config_file'),
                kwargs.get('device')
            )
        elif kwargs.get('layout_model_name') == MODEL_NAME.DocLayout_YOLO:
            atom_model = doclayout_yolo_model_init(
                kwargs.get('doclayout_yolo_weights'),
                kwargs.get('device')
            )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('mfr_cfg_path'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('ocr_show_log'),
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang')
        )
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get('table_model_name'),
            kwargs.get('table_model_path'),
            kwargs.get('table_max_time'),
            kwargs.get('device')
        )
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model
