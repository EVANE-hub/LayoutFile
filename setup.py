from pathlib import Path
from setuptools import setup, find_packages
from panda_vision.utils.version import __version__

def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    requires = [
        line.split('@')[0].strip() if "http" in line else line
        for line in lines
    ]
    return requires

# Configuration des dépendances par type d'installation
FULL_REQUIREMENTS = [
    "unimernet==0.2.1",
    "ultralytics",
    "paddlepaddle==3.0.0b1",
    "paddleocr==2.7.3",
    "paddlepaddle==3.0.0b1;platform_system=='Linux'",
    "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
    "struct-eqtable==0.3.2",
    "struct-eqtable==0.3.2",
    "einops",
    "accelerate", 
    "doclayout_yolo==0.0.2",
    "rapidocr-paddle",
    "rapid_table",
    "PyYAML",
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
]

if __name__ == '__main__':
    readme_path = Path(__file__).parent / 'README.md'
    with readme_path.open(encoding='utf-8') as file:
        long_description = file.read()

    setup(
        name="panda_vision",
        version=__version__,
        packages=find_packages() + ["panda_vision.resources"],
        package_data={"panda_vision.resources": ["**"]},
        install_requires=parse_requirements('requirements.txt'),
        extras_require={
            "full": FULL_REQUIREMENTS,
        },
        description="Un outil pratique pour convertir des PDF en Markdown",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/EVANE-hub/LayoutFile",
        python_requires=">=3.9",
        include_package_data=True,
        zip_safe=False,
    )
