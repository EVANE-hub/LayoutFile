from magic_pdf.data.data_reader_writer.multi_bucket_s3 import (
    MultiBucketS3DataReader, MultiBucketS3DataWriter)
from magic_pdf.data.schemas import S3Config


class S3DataReader(MultiBucketS3DataReader):
    def __init__(
        self,
        default_prefix_without_bucket: str,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """Client lecteur s3.

        Args:
            default_prefix_without_bucket: préfixe qui ne contient pas le bucket
            bucket (str): nom du bucket
            ak (str): clé d'accès
            sk (str): clé secrète
            endpoint_url (str): url du point de terminaison s3
            addressing_style (str, optional): Par défaut 'auto'. Les autres options valides sont 'path' et 'virtual'
            voir https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        super().__init__(
            f'{bucket}/{default_prefix_without_bucket}',
            [
                S3Config(
                    bucket_name=bucket,
                    access_key=ak,
                    secret_key=sk,
                    endpoint_url=endpoint_url,
                    addressing_style=addressing_style,
                )
            ],
        )


class S3DataWriter(MultiBucketS3DataWriter):
    def __init__(
        self,
        default_prefix_without_bucket: str,
        bucket: str,
        ak: str,
        sk: str,
        endpoint_url: str,
        addressing_style: str = 'auto',
    ):
        """Client écrivain s3.

        Args:
            default_prefix_without_bucket: préfixe qui ne contient pas le bucket
            bucket (str): nom du bucket
            ak (str): clé d'accès
            sk (str): clé secrète
            endpoint_url (str): url du point de terminaison s3
            addressing_style (str, optional): Par défaut 'auto'. Les autres options valides sont 'path' et 'virtual'
            voir https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/s3.html
        """
        super().__init__(
            f'{bucket}/{default_prefix_without_bucket}',
            [
                S3Config(
                    bucket_name=bucket,
                    access_key=ak,
                    secret_key=sk,
                    endpoint_url=endpoint_url,
                    addressing_style=addressing_style,
                )
            ],
        )
