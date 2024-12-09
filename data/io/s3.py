import boto3
from botocore.config import Config

from magic_pdf.data.io.base import IOReader, IOWriter


class S3Reader(IOReader):
    def __init__(
        self,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """Client lecteur s3.

        Args:
            bucket (str): nom du bucket
            ak (str): clé d'accès
            sk (str): clé secrète
            endpoint_url (str): url du point de terminaison s3
            addressing_style (str, optional): Par défaut 'auto'. Les autres options valides sont 'path' et 'virtual'
            voir https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        self._bucket = bucket
        self._ak = ak
        self._sk = sk
        self._s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 5, 'mode': 'standard'},
            ),
        )

    def read(self, key: str) -> bytes:
        """Lecture du fichier.

        Args:
            path (str): chemin du fichier à lire

        Returns:
            bytes: le contenu du fichier
        """
        return self.read_at(key)

    def read_at(self, key: str, offset: int = 0, limit: int = -1) -> bytes:
        """Lecture à partir d'un offset et d'une limite.

        Args:
            path (str): le chemin du fichier, si le chemin est relatif, il sera joint avec parent_dir.
            offset (int, optional): le nombre d'octets à ignorer. Par défaut 0.
            limit (int, optional): la longueur en octets à lire. Par défaut -1.

        Returns:
            bytes: le contenu du fichier
        """
        if limit > -1:
            range_header = f'bytes={offset}-{offset+limit-1}'
            res = self._s3_client.get_object(
                Bucket=self._bucket, Key=key, Range=range_header
            )
        else:
            res = self._s3_client.get_object(
                Bucket=self._bucket, Key=key, Range=f'bytes={offset}-'
            )
        return res['Body'].read()


class S3Writer(IOWriter):
    def __init__(
        self,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """Client écrivain s3.

        Args:
            bucket (str): nom du bucket
            ak (str): clé d'accès
            sk (str): clé secrète
            endpoint_url (str): url du point de terminaison s3
            addressing_style (str, optional): Par défaut 'auto'. Les autres options valides sont 'path' et 'virtual'
            voir https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        self._bucket = bucket
        self._ak = ak
        self._sk = sk
        self._s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=endpoint_url,
            config=Config(
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 5, 'mode': 'standard'},
            ),
        )

    def write(self, key: str, data: bytes):
        """Écriture du fichier avec les données.

        Args:
            path (str): le chemin du fichier, si le chemin est relatif, il sera joint avec parent_dir.
            data (bytes): les données à écrire
        """
        self._s3_client.put_object(Bucket=self._bucket, Key=key, Body=data)
