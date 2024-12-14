from panda_vision.config.enums import SupportedPdfParseMethod
from panda_vision.data.dataset import PymuDocDataset
from panda_vision.preprocessor.pdf_parser_core import PDFParser  

def parse_pdf(
    pdf_bytes,
    model_list,
    imageWriter,
    parse_mode: SupportedPdfParseMethod,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
    lang=None,
):
    dataset = PymuDocDataset(pdf_bytes)
    parser = PDFParser(model_list, dataset, imageWriter, parse_mode, lang=lang)
    return parser.parse_pdf(
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=debug_mode
    )