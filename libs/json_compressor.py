import json
import brotli
import base64

class JsonCompressor:

    @staticmethod
    def compress_json(data):
        """
        Compresse un objet json et l'encode en base64
        """
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        compressed = brotli.compress(json_bytes, quality=6)
        compressed_str = base64.b64encode(compressed).decode('utf-8')  # convertit bytes en string
        return compressed_str

    @staticmethod
    def decompress_json(compressed_str):
        """
        Décode la chaîne base64 et décompresse l'objet json
        """
        compressed = base64.b64decode(compressed_str.encode('utf-8'))  # convertit string en bytes
        decompressed_bytes = brotli.decompress(compressed)
        json_str = decompressed_bytes.decode('utf-8')
        data = json.loads(json_str)
        return data
