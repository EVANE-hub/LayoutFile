import os

from panda_vision.data.data_reader_writer.base import DataReader, DataWriter


class FileBasedDataReader(DataReader):
    def __init__(self, parent_dir: str = ''):
        """Initialisation avec parent_dir.

        Args:
            parent_dir (str, optional): le répertoire parent qui peut être utilisé dans les méthodes. Par défaut ''.
        """
        self._parent_dir = parent_dir

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Lecture à partir d'un offset et d'une limite.

        Args:
            path (str): le chemin du fichier, si le chemin est relatif, il sera joint avec parent_dir.
            offset (int, optional): le nombre d'octets à ignorer. Par défaut 0.
            limit (int, optional): la longueur en octets à lire. Par défaut -1.

        Returns:
            bytes: le contenu du fichier
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        with open(fn_path, 'rb') as f:
            f.seek(offset)
            if limit == -1:
                return f.read()
            else:
                return f.read(limit)


class FileBasedDataWriter(DataWriter):
    def __init__(self, parent_dir: str = '') -> None:
        """Initialisation avec parent_dir.

        Args:
            parent_dir (str, optional): le répertoire parent qui peut être utilisé dans les méthodes. Par défaut ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Écriture du fichier avec les données.

        Args:
            path (str): le chemin du fichier, si le chemin est relatif, il sera joint avec parent_dir.
            data (bytes): les données à écrire
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)):
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)

        with open(fn_path, 'wb') as f:
            f.write(data)
