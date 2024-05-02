import torch
from torch.nn import functional as F

from zshot import utils


def test_sorted_similarity_exact_match():
    image_features = torch.zeros(512, 512) 
    for i in range(len(image_features)):
        image_features[i,i] = 1

    # Check to see if an exact match returns
    query_exact_match_idx = 100
    query_features = image_features[query_exact_match_idx]
    indices = utils.sorted_similarity(query_features, image_features)
    assert indices[0] == query_exact_match_idx


def test_sorted_similarity_near_match():
    """Checks that a fuzzy match returns the closest result.

    Creates a set of vectors with one dimension stronger than the others.
    Checks against a query where only that same dimension has magnitude.
    """
    image_features = torch.ones(512, 512) * 0.5
    for i in range(len(image_features)):
        image_features[i,i] = 1
    image_features = F.normalize(image_features)

    query_closest_match_idx = 42
    query_features = torch.zeros(1,512) 
    query_features[0, query_closest_match_idx] = 1
    indices = utils.sorted_similarity(query_features, image_features)
    assert indices[0] == query_closest_match_idx