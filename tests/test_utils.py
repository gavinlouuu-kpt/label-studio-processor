import pytest
import numpy as np
from unittest.mock import patch, Mock
from label_studio_processor.utils import mask_to_bbox, download_image

class TestUtils:
    def test_mask_to_bbox(self, sample_mask):
        # Execute
        bbox = mask_to_bbox(sample_mask)

        # Assert
        assert bbox == (3, 2, 6, 7)  # These values correspond to the sample mask

    @patch('label_studio_processor.utils.requests.get')
    def test_download_image(self, mock_get, sample_image):
        # Setup
        mock_response = Mock()
        mock_response.content = sample_image.tobytes()
        mock_get.return_value = mock_response

        # Execute
        with patch('label_studio_processor.utils.Image.open') as mock_open:
            mock_open.return_value = sample_image
            result = download_image('http://example.com/image.jpg')

        # Assert
        assert result == sample_image
        mock_get.assert_called_once_with('http://example.com/image.jpg') 