import numpy as np
from PIL import Image
import requests
from io import BytesIO
from label_studio_sdk.converter.brush import decode_from_annotation
import logging

logger = logging.getLogger(__name__)

def mask_to_bbox(mask):
    """Convert binary mask to bounding box.
    
    Args:
        mask (numpy.ndarray): Binary mask
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def decode_mask(result):
    """Decode mask from Label Studio annotation result using the SDK.
    
    Args:
        result (dict): Annotation result from Label Studio containing:
            - type: 'brushlabels'
            - value: dict with 'rle' and 'brushlabels'
            - original_width: image width
            - original_height: image height
            
    Returns:
        numpy.ndarray: Binary mask of shape (height, width)
    """
    if not result or result.get('type') != 'brushlabels':
        logger.warning("Result is not a valid brush label annotation")
        return None
        
    try:
        # Format result for the decoder
        formatted_result = [{
            'type': 'brushlabels',
            'rle': result['value']['rle'],
            'original_width': result['original_width'],
            'original_height': result['original_height'],
            'brushlabels': result['value']['brushlabels']
        }]
        
        # Decode using label-studio-sdk
        layers = decode_from_annotation('image', formatted_result)
        
        # Get the first layer (assuming single class)
        for _, mask in layers.items():
            # Ensure mask is binary (0 or 1)
            return (mask > 0).astype(np.uint8)
            
    except Exception as e:
        logger.error(f"Error decoding mask: {str(e)}")
        return None

def download_image(url):
    """Download image from URL.
    
    Args:
        url (str): Image URL
        
    Returns:
        PIL.Image: Downloaded image
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content)) 