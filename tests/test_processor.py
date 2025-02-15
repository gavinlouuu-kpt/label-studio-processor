import pytest
from unittest.mock import patch
from label_studio_processor.processor import AnnotationProcessor

class TestAnnotationProcessor:
    def test_init(self, sample_annotation):
        processor = AnnotationProcessor(sample_annotation)
        assert processor.annotation == sample_annotation
        assert processor.result == sample_annotation['result']

    @patch('label_studio_processor.processor.download_image')
    def test_process_bbox_annotation(self, mock_download, sample_bbox_annotation, sample_image):
        # Setup
        mock_download.return_value = sample_image
        processor = AnnotationProcessor(sample_bbox_annotation)

        # Execute
        result = processor.process_annotation()

        # Assert
        assert 'bbox' in result
        assert result['bbox'] == (10, 20, 110, 70)  # x, y, x+width, y+height
        assert result['image'] == sample_image
        mock_download.assert_called_once_with('http://example.com/image.jpg')

    def test_process_bbox(self, sample_bbox_annotation):
        processor = AnnotationProcessor(sample_bbox_annotation)
        bbox = processor._process_bbox(sample_bbox_annotation['result'][0])
        assert bbox == (10, 20, 110, 70)

    @patch('label_studio_processor.processor.download_image')
    def test_process_mask_annotation(self, mock_download, sample_annotation, sample_image):
        # This test would need to be implemented once _process_mask is implemented
        pass 