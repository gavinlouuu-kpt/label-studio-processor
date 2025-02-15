import numpy as np
from PIL import Image
import requests
from io import BytesIO

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

def download_image(url):
    """Download image from URL.
    
    Args:
        url (str): Image URL
        
    Returns:
        PIL.Image: Downloaded image
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content)) 