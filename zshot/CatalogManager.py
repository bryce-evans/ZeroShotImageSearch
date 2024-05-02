"""Manager class that wraps the engine and DB, along with helpers for reading and writing to disk."""

import logging
import glob
import pickle
from typing import List
from PIL import Image
from typing import Optional

from zshot import Catalog, ZShotEngine, ZShotDatabase


class CatalogManager:
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.active_catalog_path = None
        self.active_engine = None
        self.active_db = None


    def add_image(self, path: str):
        """
        Adds a single image path to the catalog. 
        Works with local images and remote URLs.
        File handling (check if file exists) is handled by the engine.
        """
        embeddings = self.active_engine.process_images_by_paths([path])
        self.active_db.add([path], embeddings)
        self.logger.info(f"Successfully added image at {path}.\n"
                        f"Catalog Total: {self.active_db.count()}")
        

    def add_directory(self, dir: str):
        """
        Recursively adds all images in a directory to the database.
        Only supports jpg for now.
        """
        jpg_files = glob.glob(f'{dir}/**/*.jpg', recursive=True)
        embeddings = self.active_engine.process_images_by_paths(jpg_files)
        self.active_db.add(jpg_files, embeddings)
        self.logger.info(f"Successfully added {len(jpg_files)} additional images.\n"
                         f"Catalog Total: {self.active_db.count()}")
    

    def new_catalog(self, path: str):
        """Create a new catalog file.
        
        path: filename to write to.
        """
        self.active_catalog_path = path

        self.active_engine = ZShotEngine()
        self.active_db = ZShotDatabase()

        self.write_to_disk()


    def load_from_disk(self, path: str):
        """Loads a catalog from disk. 
        Will throw a FileNotFound Exception if path is not valid.
        """
        with open(path, 'rb') as f:
            catalog = pickle.load(f)
            self.active_engine = ZShotEngine(catalog.engine_name)
            self.active_db = ZShotDatabase(catalog.dimension)
            self.active_db.index = catalog.index
            self.active_catalog_path = path

            self.logger.info(f"Successfully loaded catalog {path}")


    def write_to_disk(self, path: Optional[str]=None):
        """Writes catalog data out to the active catalog path. 
        Optionally set argument `path` to create a copy in a new location.
        """
        if path == None:
            if self.active_catalog_path == None:
                self.logger.error("No active catalog.")
                return
            
            path = self.active_catalog_path
        
        catalog = Catalog(engine_name=self.active_engine.model_name, 
                          index=self.active_db.index,
                          image_paths=self.active_db.paths)

        with open(path, 'wb') as f:
            pickle.dump(catalog, f)
            self.logger.info(f"Success writing catalog {path}")
        