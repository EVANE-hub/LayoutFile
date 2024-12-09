def ocr_escape_special_markdown_char(content):
    """
    Échappe les caractères spéciaux ayant une signification particulière en markdown
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content
