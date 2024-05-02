import pytest
import torch
from torch.nn import functional as F

from zshot import ZShotDatabase, utils


def test_db_works():
    """Tests a basic insert, count, and similarity search."""
    db = ZShotDatabase()

    # Create 512 vectors with stronger magnitude
    # along one dimension as the others.
    image_features = torch.ones(512, 512) * 0.5 
    image_paths = []
    for i in range(len(image_features)):
        image_features[i,i] = 1
        image_paths.append(f"dummy_img_{i}.jpg")

    image_features = F.normalize(image_features)
        
    # Add to DB and check entries.
    db.add(image_paths, image_features)
    assert db.count() == 512

    # Create a query vector with only magnitude in one dimension.
    query_closest_match_idx = 42
    query_features = torch.zeros(1,512) 
    query_features[0, query_closest_match_idx] = 1

    # Check that the optimized DB result returns the same 
    # result as niave linear probing.
    naive_best = utils.sorted_similarity(query_features, image_features)[0]
    _, db_best = db.nearest(query_features, k=1)[0]
    assert db_best == naive_best


def test_bad_dimensions_fail():
    db = ZShotDatabase()

    bad_dimensionality = 100
    image_features = torch.ones(1, bad_dimensionality)
    image_dummy_filenames = ["img.jpg"]

    with pytest.raises(AssertionError):
        db.add(image_dummy_filenames, image_features)