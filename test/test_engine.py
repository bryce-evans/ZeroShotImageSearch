import pytest
import torch
from zshot import ZShotEngine

# TODO: This should use git lfs to avoid 3rd party dependency and network.
# Using a remote URL for the sake of the project.
# From https://stock.adobe.com/search/images?k=%22cute+cat%22
test_image = "https://t4.ftcdn.net/jpg/05/62/99/31/360_F_562993122_e7pGkeY8yMfXJcRmclsoIjtOoVDDgIlh.jpg"
test_query = "a cool cat wearing sunglasses"

def test_basic_image():
    engine = ZShotEngine()
    emb = engine.process_images([test_image])
    assert emb.shape == (1, 512)


def test_basic_query():
    engine = ZShotEngine()
    emb = engine.process_query(test_query)
    assert emb.shape == (1, 512)


def test_batch_size_working(mocker):
    engine = ZShotEngine(batch_size=8)
    mock_get_image_features = mocker.patch('transformers.models.clip.modeling_clip.CLIPModel.get_image_features')  
    
    dummy_return_emb = torch.randn(1, 512)
    mock_get_image_features.return_value = dummy_return_emb

    # Run on various batches of images and check that model is run correct 
    # number of times for the given batch size.
    # Reset the call count each time.
    engine.process_images([test_image] * 16)
    assert mock_get_image_features.call_count == 2
    mock_get_image_features.reset_mock()

    engine.process_images([test_image] * 17)
    assert mock_get_image_features.call_count == 3
    mock_get_image_features.reset_mock()

    engine.process_images([test_image] * 18)
    assert mock_get_image_features.call_count == 3
    mock_get_image_features.reset_mock()
