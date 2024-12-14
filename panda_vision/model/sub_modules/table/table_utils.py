import re


def minify_html(html):
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'\s*>\s*', '>', html)
    html = re.sub(r'\s*<\s*', '<', html)
    return html.strip()