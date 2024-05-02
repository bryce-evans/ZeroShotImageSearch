import pytest
import torch
from torch.nn import functional as F

from zshot import ZShotDatabase, utils

EMBEDDING_DIMENSION = 512

def test_db_works():
    """Tests a basic insert, count, and similarity search."""
    db = ZShotDatabase()

    # Create vectors with stronger magnitude
    # along one dimension as the others.
    EMBEDDING_COUNT = 500
    image_features = torch.ones(EMBEDDING_COUNT, EMBEDDING_DIMENSION) * 0.5 
    image_paths = []
    for i in range(len(image_features)):
        image_features[i,i] = 1
        image_paths.append(f"dummy_img_{i}.jpg")

    image_features = F.normalize(image_features)
        
    # Add to DB and check entries.
    db.add(image_paths, image_features)
    assert db.count() == EMBEDDING_COUNT

    # Create a query vector with only magnitude in one dimension.
    query_closest_match_idx = 42
    query_features = torch.zeros(1, EMBEDDING_DIMENSION) 
    query_features[0, query_closest_match_idx] = 1

    # Check that the optimized DB result returns the same 
    # result as niave linear probing.
    naive_best = utils.sorted_similarity(query_features, image_features)[0]
    indices, paths = db.nearest(query_features, k=1)
    assert indices[0] == query_closest_match_idx
    assert paths[0] == image_paths[query_closest_match_idx]


def test_bad_dimensions_fail():
    db = ZShotDatabase()

    bad_dimensionality = 100
    image_features = torch.ones(1, bad_dimensionality)
    image_dummy_filenames = ["img.jpg"]

    with pytest.raises(AssertionError):
        db.add(image_dummy_filenames, image_features)