import hashlib


def compute_md5(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest().upper()


def compute_sha256(input_string):
    hasher = hashlib.sha256()
    # Dans Python3, la chaîne doit être convertie en objet bytes pour être traitée par la fonction de hachage
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()
