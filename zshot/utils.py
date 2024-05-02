from typing import List
import torch
from PIL import Image
import requests


def sorted_similarity(query_features: torch.Tensor, image_features: torch.Tensor) -> List[int]:
    """Returns the sorted highest matches of a query to input images"""
    scores = (query_features @ image_features.T).squeeze()  
    _, indices = torch.sort(scores, descending=True)
    return indices


def image_from_url(url:str) -> Image.Image:
    data = requests.get(url, stream=True).raw
    return Image.open(data)
