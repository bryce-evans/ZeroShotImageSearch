"""Class for storing info about a photo catalog."""

import torch
from dataclasses import dataclass, field


@dataclass
class Catalog:
    path: str
    model_name: str
    vector_dims: int
    vectors: torch.Tensor = field(init=False)
    image_paths: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.vectors = torch.zeros(0, self.vector_dims)