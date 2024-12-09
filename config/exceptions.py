
class FileNotExisted(Exception):

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f'Le fichier {self.path} n\'existe pas.'


class InvalidConfig(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Configuration invalide: {self.msg}'


class InvalidParams(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Paramètres invalides: {self.msg}'


class EmptyData(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Données vides: {self.msg}'
