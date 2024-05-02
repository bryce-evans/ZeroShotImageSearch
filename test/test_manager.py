"""
Due to time, these tests are largely cut short.
Included are basic tests that show usage and reasonable proof that things work, but lots more 
can be done here in order to reduce the need for filesystem and network calls.
"""

import pytest
from PIL import Image
import numpy as np
from zshot import CatalogManager


# TODO: move this to git lfs or programmatically create an image to remove network from testing.
TEST_IMAGE_PATH = "https://t4.ftcdn.net/jpg/05/62/99/31/360_F_562993122_e7pGkeY8yMfXJcRmclsoIjtOoVDDgIlh.jpg"
TEST_QUERY = "a cool cat wearing sunglasses"


@pytest.fixture
def temp_dir(tmp_path):
    (tmp_path / "subdir").mkdir()
    yield tmp_path / "subdir"


def create_dummy_images(count=1, size=(200,200)) -> Image.Image:
    images = []
    for _ in range(count):
        data = np.random.rand(size[0], size[1], 3) * 255 
        data = data.astype(np.uint8)
        image = Image.fromarray(data, 'RGB')
        images.append(image)
    return images
        

def test_add_image_works(temp_dir):
    manager = CatalogManager()
    manager.new_catalog(temp_dir / "test.cat")
    manager.add_image(TEST_IMAGE_PATH)
    assert manager.active_db.count() == 1
    assert manager.active_db.paths[0] == TEST_IMAGE_PATH


def test_add_directory_works(temp_dir):
    images = create_dummy_images(4)
    images[0].save(temp_dir / "noise1.jpg")
    images[1].save(temp_dir / "noise2.jpg")
    images[2].save(temp_dir / "noise3.jpg")

    nested_path = (temp_dir / "recurse_test")
    nested_path.mkdir()
    images[3].save(nested_path / "noise4.jpg")

    manager = CatalogManager()
    manager.new_catalog(temp_dir / "test.cat")
    manager.add_directory(temp_dir)

    assert manager.active_db.count() == 4


def test_save_and_load_works(temp_dir):
    """
    Checks that saving and loading work from disk work.
    1. Add some results
    2. Write out to disk
    3. Clear current catalog
    4. Load and check state.
    """
    images = create_dummy_images(4)
    images[0].save(temp_dir / "noise1.jpg")
    images[1].save(temp_dir / "noise2.jpg")
    images[2].save(temp_dir / "noise3.jpg")

    nested_path = (temp_dir / "recurse_test")
    nested_path.mkdir()
    images[3].save(nested_path / "noise4.jpg")

    manager = CatalogManager()
    manager.new_catalog(temp_dir / "test.cat")
    manager.add_directory(temp_dir)

    assert manager.active_db.count() == 4
    manager.write_to_disk()

    manager.new_catalog(temp_dir / "test_2.cat")
    assert manager.active_db.count() == 0

    manager.load_from_disk(temp_dir / "test.cat")
    assert manager.active_db.count() == 4
