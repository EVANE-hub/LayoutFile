
from abc import ABC, abstractmethod


class DataReader(ABC):

    def read(self, path: str) -> bytes:
        """Lit le fichier.

        Args:
            path (str): chemin du fichier à lire

        Returns:
            bytes: le contenu du fichier
        """
        return self.read_at(path)

    @abstractmethod
    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Lit le fichier à partir d'un offset et d'une limite.

        Args:
            path (str): chemin du fichier
            offset (int, optional): nombre d'octets à ignorer. Par défaut 0.
            limit (int, optional): nombre d'octets à lire. Par défaut -1.

        Returns:
            bytes: le contenu du fichier
        """
        pass


class DataWriter(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Écrit les données dans le fichier.

        Args:
            path (str): fichier cible où écrire
            data (bytes): données à écrire
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Écrit les données dans le fichier, les données seront encodées en bytes.

        Args:
            path (str): fichier cible où écrire
            data (str): données à écrire
        """
        self.write(path, data.encode())
