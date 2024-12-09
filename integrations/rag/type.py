from enum import Enum

from pydantic import BaseModel, Field


# rag
class CategoryType(Enum):  # py310 ne supporte pas StrEnum
    text = 'text'
    title = 'title'
    interline_equation = 'interline_equation'
    image = 'image'
    image_body = 'image_body'
    image_caption = 'image_caption'
    table = 'table'
    table_body = 'table_body'
    table_caption = 'table_caption'
    table_footnote = 'table_footnote'


class ElementRelType(Enum):
    sibling = 'sibling'


class PageInfo(BaseModel):
    page_no: int = Field(description='index de la page, commence à zéro',
                         ge=0)
    height: int = Field(description='hauteur de la page', gt=0)
    width: int = Field(description='largeur de la page', ge=0)
    image_path: str | None = Field(description='image de cette page',
                                   default=None)


class ContentObject(BaseModel):
    category_type: CategoryType = Field(description='catégorie')
    poly: list[float] = Field(
        description=('Coordonnées, à convertir en coordonnées PDF,'
                     ' ordre : haut-gauche, haut-droite, bas-droite, bas-gauche'
                     ' coordonnées x,y'))
    ignore: bool = Field(description='indique si cet objet doit être ignoré',
                         default=False)
    text: str | None = Field(description='contenu textuel de l\'objet',
                             default=None)
    image_path: str | None = Field(description='chemin de l\'image intégrée',
                                   default=None)
    order: int = Field(description='ordre de cet objet dans la page',
                       default=-1)
    anno_id: int = Field(description='identifiant unique', default=-1)
    latex: str | None = Field(description='résultat latex', default=None)
    html: str | None = Field(description='résultat html', default=None)


class ElementRelation(BaseModel):
    source_anno_id: int = Field(description='identifiant unique de l\'objet source',
                                default=-1)
    target_anno_id: int = Field(description='identifiant unique de l\'objet cible',
                                default=-1)
    relation: ElementRelType = Field(
        description='relation entre l\'élément source et cible')


class LayoutElementsExtra(BaseModel):
    element_relation: list[ElementRelation] = Field(
        description='relation entre l\'élément source et cible')


class LayoutElements(BaseModel):
    layout_dets: list[ContentObject] = Field(
        description='détails des éléments de mise en page')
    page_info: PageInfo = Field(description='informations de la page')
    extra: LayoutElementsExtra = Field(description='informations supplémentaires')


# format de données itérable
class Node(BaseModel):
    category_type: CategoryType = Field(description='catégorie')
    text: str | None = Field(description='contenu textuel de l\'objet',
                             default=None)
    image_path: str | None = Field(description='chemin de l\'image intégrée',
                                   default=None)
    anno_id: int = Field(description='identifiant unique', default=-1)
    latex: str | None = Field(description='résultat latex', default=None)
    html: str | None = Field(description='résultat html', default=None)
