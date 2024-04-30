from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn import functional as F
from PIL import Image
import faiss
import numpy as np
from glob import glob
import random
import os
import logging
import sys

import argparse


logging.basicConfig(filename='clip.log', level=logging.INFO)


class ImageSearcher:

    def __init__(self, 
                 dimension=512, 
                 directory="imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/", 
                 ext="JPEG", 
                 logger=logging.getLogger(__name__)):
        self.dimension = dimension
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index = faiss.IndexFlatIP(dimension) 
        self.paths = self.load_paths(directory, ext)
        self.logger = logger
        self.batch_size = 32


    def load_paths(self, directory, extension):
        logging.error(f"Loading paths from {directory}")
        paths = glob(os.path.join(directory,f"**/*.{extension}"))
        random.shuffle(paths)
        return paths


    # images = []
    # for p in paths[:1000]:
    #     images.append(Image.open(p))

    @torch.no_grad()
    def process_images(self, paths) -> torch.Tensor:

        results = []
        for batch_idx in range(0, len(paths), self.batch_size):
            self.logger.debug(f"processing images from {batch_idx} to {min(len(paths), batch_idx + self.batch_size)}...")
            
            images = []
            for p in paths[batch_idx:batch_idx + self.batch_size]:
                img = Image.open(p)
                images.append(img)

            inputs = self.processor(images=images, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features)
            results.append(image_features)

            for img in images:
                img.close()
                
        return torch.cat(results)


    @torch.no_grad()
    def process_query(self, text_query:str) -> torch.Tensor:
        self.logger.debug(f"processing query: {text_query}")
        inputs = self.processor(text=text_query, return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features)
        return text_features


    def add_embeddings(self, embs):
        self.logger.debug(f"Adding {len(embs)} embeddings")
        self.index.add(embs)


    def get_closest(self, emb, k=4):     
        D, I = self.index.search(emb, k) 
        return I[0]

        
    def display(self, indices):
        for match in indices:
            with Image.open(self.paths[match]) as img:
                img.show()



def main(args):
    logger = logging.getLogger("mylogger")

    logging_argparse = argparse.ArgumentParser(prog=__file__, add_help=False)
    logging_argparse.add_argument('-l', '--log-level', default='DEBUG',
                                  help='set log level')
    logging_args, _ = logging_argparse.parse_known_args(args)

    try:
        logging.basicConfig(level=logging.DEBUG) #logging_args.log_level)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        logger.addHandler(ch)

    except ValueError:
        logging.error("Invalid log level: {}".format(logging_args.log_level))
        sys.exit(1)

    engine = ImageSearcher(logger=logger)

    img_embs = engine.process_images(engine.paths[:1200])
    engine.add_embeddings(img_embs)

    while True:
        query_emb = engine.process_query(input("Query: "))
        closest = engine.get_closest(query_emb)
        engine.display(closest)


if __name__ == "__main__":
    main(sys.argv[1:])