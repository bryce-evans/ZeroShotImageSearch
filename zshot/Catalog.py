"""Class for storing info about a photo catalog.

This class is explicitly a datatype for reading and writing to disk.
It is not meant to be updated in memory during runtime.
"""

import faiss
from dataclasses import dataclass, field


@dataclass
class Catalog:
    engine_name: str
    dimension: int = field(default=512)
    index: faiss.IndexFlatIP = field(default=None)
    image_paths: list = field(default_factory=list)

    def __post_init__(self):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)