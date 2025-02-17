import numpy as np
from PIL import Image
import requests
from io import BytesIO
from label_studio_sdk.converter.brush import decode_from_annotation
import logging
import os

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

def parse_annotation(annotation):
    """Parse a single Label Studio annotation to extract mask and bounding box.
    
    Args:
        annotation (dict): Label Studio annotation containing results
        
    Returns:
        tuple: (mask, bbox, class_id) where:
            - mask is a numpy array of shape (height, width)
            - bbox is a list [x_min, y_min, x_max, y_max]
            - class_id is an integer representing the class
    """
    mask = None
    bbox = None
    class_id = 0  # Default class
    
    for result in annotation['result']:
        # Get mask from brush labels
        if result['type'] == 'brushlabels':
            mask = decode_mask(result)
            # Extract class from brushlabels if available
            if result['value']['brushlabels']:
                class_id = result['value']['brushlabels'][0]  # Use first label
            
        # Get bbox from rectangle labels
        elif result['type'] == 'rectanglelabels':
            value = result['value']
            x = value['x']
            y = value['y']
            width = value['width']
            height = value['height']
            original_width = result['original_width']
            original_height = result['original_height']
            
            # Convert percentages to absolute coordinates
            x_min = int((x / 100) * original_width)
            y_min = int((y / 100) * original_height)
            x_max = int(((x + width) / 100) * original_width)
            y_max = int(((y + height) / 100) * original_height)
            
            bbox = [x_min, y_min, x_max, y_max]
            
            # Extract class from rectanglelabels if available
            if result['value']['rectanglelabels']:
                class_id = result['value']['rectanglelabels'][0]  # Use first label
    
    # If no bbox provided, compute it from mask
    if mask is not None and bbox is None:
        bbox = mask_to_bbox(mask)
        
    return mask, bbox, class_id

def prepare_training_data(label_json, images_dir):
    """Prepare training data from Label Studio JSON export.
    Only keeps samples that have valid segmentation masks.
    
    Args:
        label_json (list): List of task dictionaries from Label Studio
        images_dir (str): Path to directory containing the exported images
        
    Returns:
        dict: Dictionary containing:
            - images: Dict mapping image IDs to PIL Images
            - masks: Dict mapping image IDs to binary masks
            - box_prompts: Dict mapping image IDs to bounding boxes
            - class_ids: Dict mapping image IDs to class IDs
    """
    images = {}
    masks = {}
    box_prompts = {}
    class_ids = {}
    
    skipped_count = 0
    total_count = len(label_json)
    
    for task in label_json:
        # Skip tasks without annotations
        if not task['annotations']:
            skipped_count += 1
            continue
            
        task_id = str(task['id'])
        
        # Get first (and usually only) annotation
        annotation = task['annotations'][0]
        
        try:
            # Parse annotation to get mask, bbox and class
            mask, bbox, class_id = parse_annotation(annotation)
            
            # Only keep samples with valid masks
            if mask is None:
                logger.debug(f"Skipping task {task_id}: No valid mask found")
                skipped_count += 1
                continue
            
            # Load local image using the mapping info
            image_filename = task['file_upload']
            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                skipped_count += 1
                continue
                
            image = Image.open(image_path)
            
            images[task_id] = image
            masks[task_id] = mask
            box_prompts[task_id] = bbox
            class_ids[task_id] = class_id
                
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            skipped_count += 1
            continue
    
    logger.info(f"Prepared {len(images)} image-mask-bbox triplets")
    logger.info(f"Skipped {skipped_count} out of {total_count} tasks due to missing masks or errors")
    
    if len(images) == 0:
        logger.warning("No valid samples found in the dataset!")
    
    return {
        'images': images,
        'masks': masks,
        'box_prompts': box_prompts,
        'class_ids': class_ids
    } 