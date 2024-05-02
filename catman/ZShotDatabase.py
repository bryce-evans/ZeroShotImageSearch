import logging
import faiss
import torch
from typing import List, Tuple


class ZShotDatabase:
    """A FAISS wrapper that stores and fetches embeddings for fast lookup.

    Arguments:
        dimension: shape of the input embeddings to be stored.
        logger: logger instance to be used for logging.
    """
    def __init__(self, 
                 dimension=512, 
                 logger=logging.getLogger(__name__)):
        self.paths = []
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) 
        self.logger = logger


    def add(self, paths: List[str], embs: torch.Tensor):
        """Add rows to the database.
        Include original paths with embeddings so that we can fetch the associated image."""

        assert len(paths) == embs.shape[0], "Error: Number of paths and embeddings don't match."
        assert embs.shape[1] == self.dimension, "Error: Embedding dimensionality doesn't match database dimensionality."

        self.logger.info(f"Adding {len(embs)} embeddings")
        self.paths += paths
        self.index.add(embs)


    def count(self) -> int:
        """Returns total count of items in database."""
        return self.index.ntotal
    

    def nearest(self, emb, k=4) -> List[Tuple[str, torch.Tensor]]:
        """Takes an embedding and finds up to k matches
        returns both indices and image_paths
        """
        _, I = self.index.search(emb, k) 
        matching_indices = I[0]
        return matching_indices, [self.paths[i] for i in matching_indices]
