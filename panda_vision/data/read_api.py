import json
import os
from pathlib import Path

from panda_vision.config.exceptions import EmptyData, InvalidParams
from panda_vision.data.data_reader_writer import (FileBasedDataReader)
from panda_vision.data.dataset import ImageDataset, PymuDocDataset


def read_jsonl(
    path: str
) -> list[PymuDocDataset]:
    """Lit le fichier jsonl et retourne la liste des PymuDocDataset.

    Args:
        path (str): fichier local

    Raises:
        InvalidParams: si path est un chemin s3.
        EmptyData: si aucun emplacement de fichier pdf n'est fourni dans une ligne du fichier jsonl.

    Returns:
        list[PymuDocDataset]: chaque ligne du fichier jsonl sera convertie en PymuDocDataset
    """
    bits_arr = []
    if path.startswith('s3://'):
        raise InvalidParams('path ne doit pas être un chemin s3')
    else:
        jsonl_bits = FileBasedDataReader('').read(path)
    jsonl_d = [
        json.loads(line) for line in jsonl_bits.decode().split('\n') if line.strip()
    ]
    for d in jsonl_d:
        pdf_path = d.get('file_location', '') or d.get('path', '')
        if len(pdf_path) == 0:
            raise EmptyData("l'emplacement du fichier pdf est vide")
        if pdf_path.startswith('s3://'):
            raise InvalidParams('pdf_path ne doit pas être un chemin s3')
        else:
            bits_arr.append(FileBasedDataReader('').read(pdf_path))
    return [PymuDocDataset(bits) for bits in bits_arr]


def read_local_pdfs(path: str) -> list[PymuDocDataset]:
    """Lit les pdf depuis un chemin ou un répertoire.

    Args:
        path (str): chemin du fichier pdf ou répertoire contenant des fichiers pdf

    Returns:
        list[PymuDocDataset]: chaque fichier pdf sera converti en PymuDocDataset
    """
    if os.path.isdir(path):
        reader = FileBasedDataReader(path)
        return [
            PymuDocDataset(reader.read(doc_path.name))
            for doc_path in Path(path).glob('*.pdf')
        ]
    else:
        reader = FileBasedDataReader()
        bits = reader.read(path)
        return [PymuDocDataset(bits)]


def read_local_images(path: str, suffixes: list[str]) -> list[ImageDataset]:
    """Lit les images depuis un chemin ou un répertoire.

    Args:
        path (str): chemin du fichier image ou répertoire contenant des fichiers image
        suffixes (list[str]): les suffixes des fichiers image utilisés pour filtrer les fichiers. Exemple: ['jpg', 'png']

    Returns:
        list[ImageDataset]: chaque fichier image sera converti en ImageDataset
    """
    if os.path.isdir(path):
        imgs_bits = []
        s_suffixes = set(suffixes)
        reader = FileBasedDataReader(path)
        for root, _, files in os.walk(path):
            for file in files:
                suffix = file.split('.')
                if suffix[-1] in s_suffixes:
                    imgs_bits.append(reader.read(file))
        return [ImageDataset(bits) for bits in imgs_bits]
    else:
        reader = FileBasedDataReader()
        bits = reader.read(path)
        return [ImageDataset(bits)]
