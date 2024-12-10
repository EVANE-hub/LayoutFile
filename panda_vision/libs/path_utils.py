

def remove_non_official_s3_args(s3path):
    """
    exemple: s3://abc/xxxx.json?bytes=0,81350 ==> s3://abc/xxxx.json
    """
    arr = s3path.split("?")
    return arr[0]

def parse_s3path(s3path: str):
    # from s3pathlib import S3Path
    # p = S3Path(remove_non_official_s3_args(s3path))
    # return p.bucket, p.key
    s3path = remove_non_official_s3_args(s3path).strip()
    if s3path.startswith(('s3://', 's3a://')):
        prefix, path = s3path.split('://', 1)
        bucket_name, key = path.split('/', 1)
        return bucket_name, key
    elif s3path.startswith('/'):
        raise ValueError("Le chemin fourni commence par '/'. Cela ne correspond pas à un format de chemin S3 valide.")
    else:
        raise ValueError("Format de chemin S3 invalide. Format attendu: 's3://bucket-name/key' ou 's3a://bucket-name/key'.")


def parse_s3_range_params(s3path: str):
    """
    exemple: s3://abc/xxxx.json?bytes=0,81350 ==> [0, 81350]
    """
    arr = s3path.split("?bytes=")
    if len(arr) == 1:
        return None
    return arr[1].split(",")
