"""Class for managing a catalogue. Adding to it, fetching from it, and more.
"""

import torch

import ZShotEngine
import ZShotDatabase


class CatalogManager():
    def __init__(self, clip_model=None, dimensions=None):
        engine_kwargs = {}
        if clip_model is not None:
            engine_kwargs['model'] = clip_model

        db_kwargs = {}
        if dimensions is not None:
            db_kwargs['dimensions'] = dimensions

        self.engine = ZShotEngine(**engine_kwargs)
        self.db = ZShotDatabase(**db_kwargs)


    def write_to_disk(self, path):
        vectors = self.db.get_all_vectors()
        torch.save(vectors, path)


    def load_from_disk(self, path):
        if self.db.count() > 0:
            self.logger.warn("Loading catalogue from disk. Overwriting existing data.")
        
        self.db = ZShotDatabase
        loaded_tensor = torch.load(path)


        