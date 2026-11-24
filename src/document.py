from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Region:
    page: int
    bbox: BBox


@dataclass(frozen=True)
class MacroNode:
    id: str
    kind: Literal["section", "figure", "table"]
    text: str
    page: int
    region: Region | None = None
    section: str = ""
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mention:
    id: str
    parent: str
    text: str
    page: int
    region: Region | None = None
    kind: Literal["entity", "region", "span"] = "entity"


@dataclass
class Document:
    id: str
    pages: dict[int, Path]
    macro: tuple[MacroNode, ...] = ()
    mentions: tuple[Mention, ...] = ()
    source_pdf: Path | None = None

    def crop(self, region: Region) -> Image.Image:
        with Image.open(self.pages[region.page]) as page:
            x0, y0, x1, y1 = region.bbox
            if not (0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height):
                raise ValueError(
                    f"Invalid native bbox {region.bbox} on {self.id} page {region.page}"
                )
            return page.crop(region.bbox).convert("RGB")
