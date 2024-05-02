"""Class for managing a catalogue. Adding to it, fetching from it, and more."""

import logging
import pickle

import Catalog
import ZShotEngine
import ZShotDatabase


class CatalogManager():
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.active_catalog_path = None
        self.active_engine = None
        self.active_db = None


    def new_catalog(self, path):
        """Create a new catalog file.
        
        path: filename to write to.
        """
        self.active_catalog_path = path

        self.active_engine = ZShotEngine()
        self.active_db = ZShotDatabase()

        self.write_to_disk()


    def load_from_disk(self, path):
        """Loads a catalog from disk. 
        Will throw a FileNotFound Exception if path is not valid.
        """
        if self.active_catalog_path != None:
            self.logger.warn("Replacing existing catalog data.")

        with open(path, 'rb') as f:
            catalog_data = pickle.load(f)
            catalog = Catalog(**catalog_data)
            
            self.active_engine = ZShotEngine(catalog.engine_name)
            self.active_db = ZShotDatabase(catalog.vector_dimensions)
            self.active_db.add(catalog.image_paths, catalog.vectors)
            self.active_catalog_path = path

            self.logger.info(f"Successfully loaded catalog {path}")


    def write_to_disk(self, path=None):
        """Writes catalog data out to the active catalog path. 
        Optionally set argument `path` to create a copy in a new location.
        """
        if path == None:
            if self.active_catalog_path == None:
                self.logger.error("No active catalog.")
                return
            
            path = self.active_catalog_path
        
        catalog = Catalog(engine_name=self.active_engine.model_name, 
                          vector_dimensions=self.active_db.dimension,
                          vectors=self.active_db.get_all_vectors(),
                          image_paths=self.active_db.paths())

        with open(path, 'wb') as f:
            pickle.dump(catalog, f)
            self.logger.info("Success writing catalog {path}")




        
        


        