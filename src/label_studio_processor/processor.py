import numpy as np
from PIL import Image
from .utils import mask_to_bbox, download_image

class AnnotationProcessor:
    def __init__(self, annotation):
        """Initialize processor with an annotation.
        
        Args:
            annotation (dict): Annotation from Label Studio
        """
        self.annotation = annotation
        self.result = annotation.get('result', [])
        self.task = annotation.get('task', {})
    
    def process_annotation(self):
        """Process the annotation to extract mask and bbox.
        
        Returns:
            dict: Processed data containing original image, mask, and bbox
        """
        processed_data = {
            'image_url': self.task.get('data', {}).get('image'),
            'task_id': self.task.get('id'),
            'annotations': []
        }
        
        for item in self.result:
            annotation_type = item.get('type')
            if annotation_type == 'brushlabels':
                mask_data = self._process_mask(item)
                processed_data['annotations'].append({
                    'type': 'mask',
                    'label': item.get('value', {}).get('brushlabels', [])[0],
                    'mask': mask_data
                })
            elif annotation_type == 'rectanglelabels':
                bbox_data = self._process_bbox(item)
                processed_data['annotations'].append({
                    'type': 'bbox',
                    'label': item.get('value', {}).get('rectanglelabels', [])[0],
                    'bbox': bbox_data
                })
        
        return processed_data
    
    def _process_mask(self, item):
        """Process segmentation mask data.
        
        Args:
            item (dict): Mask annotation item
            
        Returns:
            numpy.ndarray: Binary mask
        """
        # Implementation depends on Label Studio's mask format
        # Will implement once we see the actual mask data structure
        return item.get('value', {})
    
    def _process_bbox(self, item):
        """Process bounding box data.
        
        Args:
            item (dict): Bbox annotation item
            
        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        value = item.get('value', {})
        return {
            'x': value.get('x'),
            'y': value.get('y'),
            'width': value.get('width'),
            'height': value.get('height')
        } 