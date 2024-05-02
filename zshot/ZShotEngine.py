import logging
from typing import List, Union

from PIL import Image
import requests
import torch
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor


class ZShotEngine:
    """CLIP wrapper that takes images and text inputs and maps to a common embedding space.

    Arguments:
        model: CLIP backbone model name. Additional model options can be found at https://huggingface.co/models?other=clip
        batch_size: number of images to process at once. 
        logger: logger instance to be used for logging.
    """
    def __init__(self, 
                 model="openai/clip-vit-base-patch32",
                 batch_size=32,
                 logger=logging.getLogger(__name__)):
        
        self.model_name = model
        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)
        self.batch_size = batch_size
        self.logger = logger
        


    @torch.no_grad()
    def process_image_batch(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """A helper for process_images that takes a single batch of image data"""
        inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features)
        return image_features
        

    def process_images_by_paths(self, paths: List[str]) -> torch.Tensor:
        """
        Process a set of image paths and return a single tensor of embedding results. 
        Paths can be remote URLs starting with https://
        """
        results = []
        for batch_idx in range(0, len(paths), self.batch_size):
            self.logger.info(f"Processing images from {batch_idx} to {min(len(paths), batch_idx + self.batch_size)}...")
            
            images = []
            for p in paths[batch_idx:batch_idx + self.batch_size]:

                # Convert remote links to loadable data
                if p.startswith("https://"):
                    p = requests.get(p, stream=True).raw

                img = Image.open(p)
                images.append(img)

            image_features = self.process_image_batch(images)
            results.append(image_features)

            for img in images:
                img.close()
                
        return torch.cat(results)
    

    @torch.no_grad()
    def process_query(self, text_query:str) -> torch.Tensor:
        """Processes a query string and return a tensor of the result embedding."""
        self.logger.debug(f"Processing query: \"{text_query}\"")
        inputs = self.processor(text=text_query, return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features)
        return text_features
