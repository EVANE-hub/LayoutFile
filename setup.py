from pathlib import Path
from setuptools import setup, find_packages
from panda_vision.libs.version import __version__


def parse_requirements(filename):
    """Parse le fichier requirements.txt et retourne la liste des dépendances."""
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


# Configuration des dépendances par type d'installation
LITE_REQUIREMENTS = [
    "paddleocr==2.7.3",
    "paddlepaddle==3.0.0b1;platform_system=='Linux'",
    "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
]

FULL_REQUIREMENTS = [
    "unimernet==0.2.1",
    "matplotlib<=3.9.0;platform_system=='Windows'",
    "matplotlib;platform_system=='Linux' or platform_system=='Darwin'",
    "ultralytics",
    "paddleocr==2.7.3",
    "paddlepaddle==3.0.0b1;platform_system=='Linux'",
    "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
    "struct-eqtable==0.3.2",
    "einops",
    "accelerate", 
    "doclayout_yolo==0.0.2",
    "rapidocr-paddle",
    "rapid_table",
    "PyYAML",
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
]

OLD_LINUX_REQUIREMENTS = [
    "albumentations<=1.4.20"
]

if __name__ == '__main__':
    # Lecture du README
    readme_path = Path(Path(__file__).parent, 'README.md')
    with readme_path.open(encoding='utf-8') as file:
        long_description = file.read()

    setup(
        name="panda_vision",
        version=__version__,
        packages=find_packages() + ["panda_vision.resources"],
        package_data={
            "panda_vision.resources": ["**"],
        },
        install_requires=parse_requirements('requirements.txt'),
        extras_require={
            "lite": LITE_REQUIREMENTS,
            "full": FULL_REQUIREMENTS,
            "old_linux": OLD_LINUX_REQUIREMENTS
        },
        description="Un outil pratique pour convertir des PDF en Markdown",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/EVANE-hub/LayoutFile",
        python_requires=">=3.9",
        entry_points={
            "console_scripts": [
                "panda-vision = panda_vision.tools.cli:cli",
                "panda-vision-dev = panda_vision.tools.cli_dev:cli"
            ],
        },
        include_package_data=True,
        zip_safe=False,
    )
