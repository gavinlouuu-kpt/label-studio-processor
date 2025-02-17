import numpy as np
from PIL import Image
import requests
from io import BytesIO
from label_studio_sdk.converter.brush import decode_from_annotation
import logging
import os
from tqdm import tqdm

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

def create_class_mapping(label_json):
    """Create a mapping of class names to class IDs from all annotations.
    
    Args:
        label_json (list): List of task dictionaries from Label Studio
        
    Returns:
        dict: Mapping from class names to integer IDs
    """
    unique_classes = set()
    
    # Collect all unique class names
    for task in label_json:
        if not task['annotations']:
            continue
            
        for annotation in task['annotations']:
            for result in annotation['result']:
                if result['type'] == 'brushlabels' and result['value'].get('brushlabels'):
                    unique_classes.update(result['value']['brushlabels'])
                elif result['type'] == 'rectanglelabels' and result['value'].get('rectanglelabels'):
                    unique_classes.update(result['value']['rectanglelabels'])
    
    # Create mapping (sorted to ensure consistent IDs)
    class_map = {class_name: idx for idx, class_name in enumerate(sorted(unique_classes))}
    
    logger.info(f"Found {len(class_map)} unique classes: {class_map}")
    return class_map

def parse_annotation(annotation, class_map):
    """Parse a single Label Studio annotation to extract mask and bounding box.
    
    Args:
        annotation (dict): Label Studio annotation containing results
        class_map (dict): Mapping from class names to class IDs
        
    Returns:
        tuple: (mask, bbox, class_id, status) where:
            - mask is a numpy array of shape (height, width)
            - bbox is a list [x_min, y_min, x_max, y_max]
            - class_id is an integer representing the class
            - status is a string indicating any issues ('ok', 'no_class', 'unknown_class')
    """
    mask = None
    bbox = None
    class_name = None  # Store class name first
    
    for result in annotation['result']:
        # Get mask from brush labels
        if result['type'] == 'brushlabels':
            mask = decode_mask(result)
            # Extract class from brushlabels if available
            if result['value'].get('brushlabels'):
                class_name = result['value']['brushlabels'][0]  # Use first label
            
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
            if result['value'].get('rectanglelabels'):
                class_name = result['value']['rectanglelabels'][0]  # Use first label
    
    # If no bbox provided, compute it from mask
    if mask is not None and bbox is None:
        bbox = mask_to_bbox(mask)
    
    # Convert class name to ID
    if class_name is None:
        class_id = 0
        status = 'no_class'
    else:
        if class_name not in class_map:
            class_id = 0
            status = 'unknown_class'
        else:
            class_id = class_map[class_name]
            status = 'ok'
        
    return mask, bbox, class_id, status

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
            - class_map: Dict mapping class names to class IDs
    """
    # Create class mapping first
    class_map = create_class_mapping(label_json)
    
    images = {}
    masks = {}
    box_prompts = {}
    class_ids = {}
    
    # Initialize counters
    total_count = len(label_json)
    no_annotation_count = total_count - len([task for task in label_json if task.get('annotations')])
    no_class_count = 0
    unknown_class_count = 0
    no_mask_count = 0
    missing_image_count = 0
    error_count = 0
    
    # Filter tasks with annotations first
    valid_tasks = [task for task in label_json if task.get('annotations')]
    if no_annotation_count > 0:
        logger.info(f"Skipping {no_annotation_count} tasks without annotations")
    
    # Process each task with progress bar
    for task in tqdm(valid_tasks, desc="Processing annotations"):
        task_id = str(task['id'])
        
        # Get first (and usually only) annotation
        annotation = task['annotations'][0]
        
        try:
            # Parse annotation to get mask, bbox and class
            mask, bbox, class_id, status = parse_annotation(annotation, class_map)
            
            # Track status
            if status == 'no_class':
                no_class_count += 1
            elif status == 'unknown_class':
                unknown_class_count += 1
            
            # Only keep samples with valid masks
            if mask is None:
                no_mask_count += 1
                continue
            
            # Load local image using the mapping info
            image_filename = task['file_upload']
            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                missing_image_count += 1
                continue
                
            image = Image.open(image_path)
            
            images[task_id] = image
            masks[task_id] = mask
            box_prompts[task_id] = bbox
            class_ids[task_id] = class_id
                
        except Exception as e:
            error_count += 1
            continue
    
    # Log summary
    logger.info(f"Successfully prepared {len(images)} image-mask-bbox triplets")
    if no_annotation_count + no_class_count + unknown_class_count + no_mask_count + missing_image_count + error_count > 0:
        logger.info("Skipped tasks summary:")
        if no_annotation_count > 0:
            logger.info(f"- {no_annotation_count} without annotations")
        if no_class_count > 0:
            logger.info(f"- {no_class_count} without class labels")
        if unknown_class_count > 0:
            logger.info(f"- {unknown_class_count} with unknown classes")
        if no_mask_count > 0:
            logger.info(f"- {no_mask_count} without valid masks")
        if missing_image_count > 0:
            logger.info(f"- {missing_image_count} with missing images")
        if error_count > 0:
            logger.info(f"- {error_count} with processing errors")
    
    if len(images) == 0:
        logger.warning("No valid samples found in the dataset!")
    
    return {
        'images': images,
        'masks': masks,
        'box_prompts': box_prompts,
        'class_ids': class_ids,
        'class_map': class_map
    }

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert (x_min, y_min, x_max, y_max) to YOLO format (x_center, y_center, width, height).
    All values are normalized to [0, 1].
    
    Args:
        bbox (tuple): Bounding box in format (x_min, y_min, x_max, y_max)
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate center points
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize to [0, 1]
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return x_center, y_center, width, height 