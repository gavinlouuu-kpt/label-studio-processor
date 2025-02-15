import pytest
import numpy as np
from PIL import Image

@pytest.fixture
def sample_annotation():
    return {
        'result': [
            {
                'type': 'brushlabels',
                'value': {
                    'brushlabels': ['class1'],
                    'rle': {'counts': [0, 5, 5, 5], 'size': [10, 10]}  # Simplified RLE
                }
            }
        ],
        'data': {
            'image': 'http://example.com/image.jpg'
        }
    }

@pytest.fixture
def sample_bbox_annotation():
    return {
        'result': [
            {
                'type': 'rectanglelabels',
                'value': {
                    'x': 10,
                    'y': 20,
                    'width': 100,
                    'height': 50,
                    'rectanglelabels': ['class1']
                }
            }
        ],
        'data': {
            'image': 'http://example.com/image.jpg'
        }
    }

@pytest.fixture
def sample_mask():
    # Create a 10x10 binary mask with a rectangle in the middle
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 3:7] = True
    return mask

@pytest.fixture
def sample_image():
    # Create a small test image
    return Image.new('RGB', (100, 100), color='red') 