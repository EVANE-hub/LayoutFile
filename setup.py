from pathlib import Path
from setuptools import setup, find_packages
from panda_vision.libs.version import __version__


def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    requires = []

    for line in lines:
        if "http" in line:
            pkg_name_without_url = line.split('@')[0].strip()
            requires.append(pkg_name_without_url)
        else:
            requires.append(line)

    return requires


if __name__ == '__main__':
    with Path(Path(__file__).parent,
              'README.md').open(encoding='utf-8') as file:
        long_description = file.read()
    setup(  
        name="panda_vision",  # nom du projet
        version=__version__,  # obtient automatiquement le numéro de version à partir du tag
        packages=find_packages() + ["panda_vision.resources"],  # inclut tous les packages
        package_data={
            "panda_vision.resources": ["**"],  # inclut tous les fichiers dans le répertoire panda_vision.resources
        },
        install_requires=parse_requirements('requirements.txt'),  # dépendances tierces du projet
        extras_require={
            "lite": ["paddleocr==2.7.3",
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
                     ],
            "full": ["unimernet==0.2.1",  # mise à niveau unimernet vers 0.2.1
                     "matplotlib<=3.9.0;platform_system=='Windows'",  # 3.9.1 et versions ultérieures ne fournissent pas de packages précompilés pour Windows, évite les échecs d'installation sur les machines Windows sans environnement de compilation
                     "matplotlib;platform_system=='Linux' or platform_system=='Darwin'",  # linux et macos ne devraient pas limiter la version maximale de matplotlib pour éviter les bugs dus à l'impossibilité de mise à jour
                     "ultralytics",  # yolov8, détection de formules
                     "paddleocr==2.7.3",  # les versions 2.8.0 et 2.8.1 sont en conflit avec detectron2, nécessite de fixer à 2.7.3
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",  # résout les problèmes de segmentation sur Linux
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",  # la version 3.0.0b1 pour Windows a une baisse de performance, nécessite de fixer à 2.6.1
                     "struct-eqtable==0.3.2",  # analyse de tableaux
                     "einops",  # dépendance de struct-eqtable
                     "accelerate",  # dépendance de struct-eqtable
                     "doclayout_yolo==0.0.2",  # doclayout_yolo
                     "rapidocr-paddle",  # rapidocr-paddle
                     "rapid_table",  # rapid_table
                     "PyYAML",  # yaml
                     "detectron2"
                     ],
            "old_linux":[
                "albumentations<=1.4.20", # simsimd introduit dans 1.4.21 n'est pas compatible avec les systèmes Linux de 2019 et antérieurs
            ]
        },
        description="Un outil pratique pour convertir des PDF en Markdown",  # description courte
        long_description=long_description,  # description détaillée
        long_description_content_type="text/markdown",  # si README est au format Markdown
        url="https://github.com/EVANE-hub/LayoutFile",
        python_requires=">=3.9",  # version Python requise pour le projet
        entry_points={
            "console_scripts": [
                "panda-vision = panda_vision.tools.cli:cli",
                "panda-vision-dev = panda_vision.tools.cli_dev:cli" 
            ],
        },  # commandes exécutables fournies par le projet
        include_package_data=True,  # inclure ou non les fichiers non-code comme les fichiers de données, de configuration, etc.
        zip_safe=False,  # utiliser ou non le format zip pour l'empaquetage, généralement False
    )
