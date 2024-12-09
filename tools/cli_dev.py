import json as json_parse
import os
from pathlib import Path

import click

import magic_pdf.model as model_config
from magic_pdf.data.data_reader_writer import FileBasedDataReader, S3DataReader
from magic_pdf.libs.config_reader import get_s3_config
from magic_pdf.libs.path_utils import (parse_s3_range_params, parse_s3path,
                                       remove_non_official_s3_args)
from magic_pdf.libs.version import __version__
from magic_pdf.tools.common import do_parse, parse_pdf_methods


def read_s3_path(s3path):
    bucket, key = parse_s3path(s3path)

    s3_ak, s3_sk, s3_endpoint = get_s3_config(bucket)
    s3_rw = S3DataReader('', bucket, s3_ak, s3_sk, s3_endpoint, 'auto')
    may_range_params = parse_s3_range_params(s3path)
    if may_range_params is None or 2 != len(may_range_params):
        byte_start, byte_end = 0, -1
    else:
        byte_start, byte_end = int(may_range_params[0]), int(
            may_range_params[1])
    return s3_rw.read_at(
        remove_non_official_s3_args(s3path),
        byte_start,
        byte_end,
    )


@click.group()
@click.version_option(__version__, '--version', '-v', help='Afficher les informations de version')
def cli():
    pass


@cli.command()
@click.option(
    '-j',
    '--jsonl',
    'jsonl',
    type=str,
    help='Chemin d\'entrée jsonl, fichier local ou sur s3',
    required=True,
)
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help='Spécifier la méthode d\'analyse. txt: méthode d\'analyse PDF textuel, ocr: analyse par reconnaissance optique, auto: sélection intelligente de la méthode',
    default='auto',
)
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(),
    required=True,
    help='Répertoire de sortie local',
)
def jsonl(jsonl, method, output_dir):
    model_config.__use_inside_model__ = False
    if jsonl.startswith('s3://'):
        jso = json_parse.loads(read_s3_path(jsonl).decode('utf-8'))
    else:
        with open(jsonl) as f:
            jso = json_parse.loads(f.readline())
    os.makedirs(output_dir, exist_ok=True)
    s3_file_path = jso.get('file_location')
    if s3_file_path is None:
        s3_file_path = jso.get('path')
    pdf_file_name = Path(s3_file_path).stem
    pdf_data = read_s3_path(s3_file_path)

    print(pdf_file_name, jso, method)
    do_parse(
        output_dir,
        pdf_file_name,
        pdf_data,
        jso['doc_layout_result'],
        method,
        False,
        f_dump_content_list=True,
        f_draw_model_bbox=True,
    )


@cli.command()
@click.option(
    '-p',
    '--pdf',
    'pdf',
    type=click.Path(exists=True),
    required=True,
    help='Fichier PDF local',
)
@click.option(
    '-j',
    '--json',
    'json_data',
    type=click.Path(exists=True),
    required=True,
    help='Données JSON inférées par le modèle local',
)
@click.option('-o',
              '--output-dir',
              'output_dir',
              type=click.Path(),
              required=True,
              help='Répertoire de sortie local')
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help='Spécifier la méthode d\'analyse. txt: méthode d\'analyse PDF textuel, ocr: analyse par reconnaissance optique, auto: sélection intelligente de la méthode',
    default='auto',
)
def pdf(pdf, json_data, output_dir, method):
    model_config.__use_inside_model__ = False
    full_pdf_path = os.path.realpath(pdf)
    os.makedirs(output_dir, exist_ok=True)

    def read_fn(path):
        disk_rw = FileBasedDataReader(os.path.dirname(path))
        return disk_rw.read(os.path.basename(path))

    model_json_list = json_parse.loads(read_fn(json_data).decode('utf-8'))

    file_name = str(Path(full_pdf_path).stem)
    pdf_data = read_fn(full_pdf_path)
    do_parse(
        output_dir,
        file_name,
        pdf_data,
        model_json_list,
        method,
        False,
        f_dump_content_list=True,
        f_draw_model_bbox=True,
    )


if __name__ == '__main__':
    cli()