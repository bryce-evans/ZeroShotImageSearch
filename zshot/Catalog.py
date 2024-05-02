"""Class for storing info about a photo catalog.

This class is explicitly a datatype for reading and writing to disk.
It is not meant to be updated in memory during runtime.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class Catalog:
    engine_name: str
    vector_dimensions: int
    vectors: torch.Tensor = field(init=False)
    image_paths: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.vectors = torch.zeros(0, self.vector_dimensions)