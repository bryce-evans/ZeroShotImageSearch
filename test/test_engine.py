import torch

from torchvision.transforms import ToPILImage

from catman import ZShotEngine, utils


MODEL_INPUT_SHAPE = torch.Size((3, 224, 224))
MODEL_OUTPUT_DIMENSION = 512

# TODO: This should use git lfs to avoid 3rd party dependency and network.
# Using a remote URL for the sake of the project.
# From https://stock.adobe.com/search/images?k=%22cute+cat%22
TEST_IMAGE_PATH = "https://t4.ftcdn.net/jpg/05/62/99/31/360_F_562993122_e7pGkeY8yMfXJcRmclsoIjtOoVDDgIlh.jpg"
TEST_QUERY = "a cool cat wearing sunglasses"


def test_basic_image():
    engine = ZShotEngine()
    emb = engine.process_images_by_paths([TEST_IMAGE_PATH])
    assert emb.shape == (1, MODEL_OUTPUT_DIMENSION)


def test_basic_query():
    engine = ZShotEngine()
    emb = engine.process_query(TEST_QUERY)
    assert emb.shape == (1, MODEL_OUTPUT_DIMENSION)


def test_various_input_sizes_work():
    engine = ZShotEngine()

    # Exact expected input size works.
    input = torch.ones(MODEL_INPUT_SHAPE)
    result = engine.process_image_batch(input)
    assert result.shape == (1, MODEL_OUTPUT_DIMENSION)

    # Small image works.
    input = torch.ones(3, 100, 100)
    result = engine.process_image_batch(input)
    assert result.shape == (1, MODEL_OUTPUT_DIMENSION)

    # Large image works.
    input = torch.ones(3, 1000, 1000)
    result = engine.process_image_batch(input)
    assert result.shape == (1, MODEL_OUTPUT_DIMENSION)

    # Different aspect ratio works.
    input = torch.ones(3, 100, 1000)
    result = engine.process_image_batch(input)
    assert result.shape == (1, MODEL_OUTPUT_DIMENSION)


def test_model_returns_reasonable_result():
    """Tests if a query matches a photo closer than random noise."""
    torch.random.manual_seed(0xad0be)

    # Position of real photo in the batch.
    real_photo_index = 9
    
    # Creates a batch of noise images of size 224x224.
    # Then insert the real image in the middle.
    to_pil = ToPILImage()
    images = [to_pil(torch.rand(MODEL_INPUT_SHAPE)) for _ in range(31)]
    real_photo = utils.image_from_url(TEST_IMAGE_PATH)
    images.insert(real_photo_index, real_photo)

    engine = ZShotEngine()
    images_features = engine.process_image_batch(images)
    query_features = engine.process_query(TEST_QUERY)

    indices = utils.sorted_similarity(query_features, images_features)

    # Check that the best match is the photo that was inserted above.
    assert indices[0] == real_photo_index


def test_batch_size_working(mocker):
    engine = ZShotEngine(batch_size=8)
    mock_get_image_features = mocker.patch('transformers.models.clip.modeling_clip.CLIPModel.get_image_features')  
    
    dummy_return_emb = torch.randn(1, MODEL_OUTPUT_DIMENSION)
    mock_get_image_features.return_value = dummy_return_emb

    # Run on various batches of images and check that model is run correct 
    # number of times for the given batch size.
    # Reset the call count each time.
    engine.process_images_by_paths([TEST_IMAGE_PATH] * 16)
    assert mock_get_image_features.call_count == 2
    mock_get_image_features.reset_mock()

    engine.process_images_by_paths([TEST_IMAGE_PATH] * 17)
    assert mock_get_image_features.call_count == 3
    mock_get_image_features.reset_mock()

    engine.process_images_by_paths([TEST_IMAGE_PATH] * 18)
    assert mock_get_image_features.call_count == 3
    mock_get_image_features.reset_mock()
