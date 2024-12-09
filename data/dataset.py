from abc import ABC, abstractmethod
from typing import Iterator

import fitz

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.schemas import PageInfo
from magic_pdf.data.utils import fitz_doc_to_image


class PageableData(ABC):
    @abstractmethod
    def get_image(self) -> dict:
        """Transforme les données en image."""
        pass

    @abstractmethod
    def get_doc(self) -> fitz.Page:
        """Obtient la page pymudoc."""
        pass

    @abstractmethod
    def get_page_info(self) -> PageInfo:
        """Obtient les informations de la page.

        Returns:
            PageInfo: les informations de cette page
        """
        pass


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """La longueur du jeu de données."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[PageableData]:
        """Renvoie les données de la page."""
        pass

    @abstractmethod
    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """Les méthodes supportées par ce jeu de données.

        Returns:
            list[SupportedPdfParseMethod]: Les méthodes supportées, les méthodes valides sont : OCR, TXT
        """
        pass

    @abstractmethod
    def data_bits(self) -> bytes:
        """Les bits utilisés pour créer ce jeu de données."""
        pass

    @abstractmethod
    def get_page(self, page_id: int) -> PageableData:
        """Obtient la page indexée par page_id.

        Args:
            page_id (int): l'index de la page

        Returns:
            PageableData: l'objet doc de la page
        """
        pass


class PymuDocDataset(Dataset):
    def __init__(self, bits: bytes):
        """Initialise le jeu de données qui encapsule les documents pymudoc.

        Args:
            bits (bytes): les octets du pdf
        """
        self._records = [Doc(v) for v in fitz.open('pdf', bits)]
        self._data_bits = bits
        self._raw_data = bits

    def __len__(self) -> int:
        """Le nombre de pages du pdf."""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """Renvoie l'objet doc de la page."""
        return iter(self._records)

    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """La méthode supportée par ce jeu de données.

        Returns:
            list[SupportedPdfParseMethod]: les méthodes supportées
        """
        return [SupportedPdfParseMethod.OCR, SupportedPdfParseMethod.TXT]

    def data_bits(self) -> bytes:
        """Les bits pdf utilisés pour créer ce jeu de données."""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """L'objet doc de la page.

        Args:
            page_id (int): l'index du doc de la page

        Returns:
            PageableData: l'objet doc de la page
        """
        return self._records[page_id]


class ImageDataset(Dataset):
    def __init__(self, bits: bytes):
        """Initialise le jeu de données qui encapsule les documents pymudoc.

        Args:
            bits (bytes): les octets de la photo qui sera d'abord convertie en pdf, puis en pymudoc.
        """
        pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
        self._records = [Doc(v) for v in fitz.open('pdf', pdf_bytes)]
        self._raw_data = bits
        self._data_bits = pdf_bytes

    def __len__(self) -> int:
        """La longueur du jeu de données."""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """Renvoie l'objet de la page."""
        return iter(self._records)

    def supported_methods(self):
        """La méthode supportée par ce jeu de données.

        Returns:
            list[SupportedPdfParseMethod]: les méthodes supportées
        """
        return [SupportedPdfParseMethod.OCR]

    def data_bits(self) -> bytes:
        """Les bits pdf utilisés pour créer ce jeu de données."""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """L'objet doc de la page.

        Args:
            page_id (int): l'index du doc de la page

        Returns:
            PageableData: l'objet doc de la page
        """
        return self._records[page_id]


class Doc(PageableData):
    """Initialisé avec l'objet pymudoc."""
    def __init__(self, doc: fitz.Page):
        self._doc = doc

    def get_image(self):
        """Renvoie les informations de l'image.

        Returns:
            dict: {
                img: np.ndarray,
                width: int,
                height: int
            }
        """
        return fitz_doc_to_image(self._doc)

    def get_doc(self) -> fitz.Page:
        """Obtient l'objet pymudoc.

        Returns:
            fitz.Page: l'objet pymudoc
        """
        return self._doc

    def get_page_info(self) -> PageInfo:
        """Obtient les informations de la page.

        Returns:
            PageInfo: les informations de cette page
        """
        page_w = self._doc.rect.width
        page_h = self._doc.rect.height
        return PageInfo(w=page_w, h=page_h)

    def __getattr__(self, name):
        if hasattr(self._doc, name):
            return getattr(self._doc, name)
