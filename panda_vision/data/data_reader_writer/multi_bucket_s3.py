import os
from panda_vision.config.exceptions import InvalidConfig, InvalidParams
from panda_vision.data.data_reader_writer.base import DataReader, DataWriter
from panda_vision.data.io.s3 import S3Reader, S3Writer
from panda_vision.data.schemas import S3Config
from panda_vision.libs.path_utils import (parse_s3_range_params, parse_s3path, remove_non_official_s3_args)


class MultiS3Mixin:
    def __init__(self, default_prefix: str, s3_configs: list[S3Config]):
        """Initialisation avec plusieurs configurations s3.

        Args:
            default_prefix (str): le préfixe par défaut du chemin relatif. Par exemple, {some_bucket}/{some_prefix} ou {some_bucket}
            s3_configs (list[S3Config]): liste des configurations s3, le bucket_name doit être unique dans la liste.

        Raises:
            InvalidConfig: la configuration du bucket par défaut n'est pas dans s3_configs.
            InvalidConfig: le nom du bucket n'est pas unique dans s3_configs.
            InvalidConfig: le bucket par défaut doit être fourni.
        """
        if len(default_prefix) == 0:
            raise InvalidConfig('default_prefix doit être fourni')
    
        arr = default_prefix.strip("/").split("/")
        self.default_bucket = arr[0]
        self.default_prefix = "/".join(arr[1:])

        found_default_bucket_config = False
        for conf in s3_configs:
            if conf.bucket_name == self.default_bucket:
                found_default_bucket_config = True
                break

        if not found_default_bucket_config:
            raise InvalidConfig(
                f'default_bucket: {self.default_bucket} la configuration doit être fournie dans s3_configs: {s3_configs}'
            )

        uniq_bucket = set([conf.bucket_name for conf in s3_configs])
        if len(uniq_bucket) != len(s3_configs):
            raise InvalidConfig(
                f'le bucket_name dans s3_configs: {s3_configs} doit être unique'
            )

        self.s3_configs = s3_configs
        self._s3_clients_h: dict = {}


class MultiBucketS3DataReader(DataReader, MultiS3Mixin):
    def read(self, path: str) -> bytes:
        """Lit le chemin depuis s3, sélectionne différents clients bucket pour chaque requête
        basé sur le bucket, supporte aussi la lecture par plage.

        Args:
            path (str): le chemin s3 du fichier, le chemin doit être au format s3://bucket_name/path?offset,limit.
            par exemple: s3://bucket_name/path?0,100.

        Returns:
            bytes: le contenu du fichier s3.
        """
        may_range_params = parse_s3_range_params(path)
        if may_range_params is None or 2 != len(may_range_params):
            byte_start, byte_len = 0, -1
        else:
            byte_start, byte_len = int(may_range_params[0]), int(may_range_params[1])
        path = remove_non_official_s3_args(path)
        return self.read_at(path, byte_start, byte_len)

    def __get_s3_client(self, bucket_name: str):
        if bucket_name not in set([conf.bucket_name for conf in self.s3_configs]):
            raise InvalidParams(
                f'nom du bucket: {bucket_name} non trouvé dans s3_configs: {self.s3_configs}'
            )
        if bucket_name not in self._s3_clients_h:
            conf = next(
                filter(lambda conf: conf.bucket_name == bucket_name, self.s3_configs)
            )
            self._s3_clients_h[bucket_name] = S3Reader(
                bucket_name,
                conf.access_key,
                conf.secret_key,
                conf.endpoint_url,
                conf.addressing_style,
            )
        return self._s3_clients_h[bucket_name]

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Lit le fichier avec offset et limit, sélectionne différents clients bucket
        pour chaque requête basé sur le bucket.

        Args:
            path (str): le chemin du fichier.
            offset (int, optional): le nombre d'octets à ignorer. Par défaut 0.
            limit (int, optional): le nombre d'octets à lire. Par défaut -1 ce qui signifie infini.

        Returns:
            bytes: le contenu du fichier.
        """
        if path.startswith('s3://'):
            bucket_name, path = parse_s3path(path)
            s3_reader = self.__get_s3_client(bucket_name)
        else:
            s3_reader = self.__get_s3_client(self.default_bucket)
            path = os.path.join(self.default_prefix, path)
        return s3_reader.read_at(path, offset, limit)


class MultiBucketS3DataWriter(DataWriter, MultiS3Mixin):
    def __get_s3_client(self, bucket_name: str):
        if bucket_name not in set([conf.bucket_name for conf in self.s3_configs]):
            raise InvalidParams(
                f'nom du bucket: {bucket_name} non trouvé dans s3_configs: {self.s3_configs}'
            )
        if bucket_name not in self._s3_clients_h:
            conf = next(
                filter(lambda conf: conf.bucket_name == bucket_name, self.s3_configs)
            )
            self._s3_clients_h[bucket_name] = S3Writer(
                bucket_name,
                conf.access_key,
                conf.secret_key,
                conf.endpoint_url,
                conf.addressing_style,
            )
        return self._s3_clients_h[bucket_name]

    def write(self, path: str, data: bytes) -> None:
        """Écrit le fichier avec les données, sélectionne aussi différents clients bucket
        pour chaque requête basé sur le bucket.

        Args:
            path (str): le chemin du fichier, si le chemin est relatif, il sera joint avec parent_dir.
            data (bytes): les données à écrire.
        """
        if path.startswith('s3://'):
            bucket_name, path = parse_s3path(path)
            s3_writer = self.__get_s3_client(bucket_name)
        else:
            s3_writer = self.__get_s3_client(self.default_bucket)
            path = os.path.join(self.default_prefix, path)
        return s3_writer.write(path, data)
