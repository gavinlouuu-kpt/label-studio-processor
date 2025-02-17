"""Functions for exporting data from Label Studio."""

import os
import json
import logging
import requests
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
import shutil
from PIL import Image
from .client import LabelStudioClient
from .utils import decode_mask, mask_to_bbox
from .data import load_label_studio_data

logger = logging.getLogger(__name__)

def download_file(url, output_path, headers=None):
    """Download a file from URL to the specified path with progress bar.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        headers (dict, optional): Headers for the request
    """
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    # Get file size for progress bar
    file_size = int(response.headers.get('content-length', 0))
    
    # Use tqdm for download progress
    progress = tqdm(
        total=file_size,
        unit='iB',
        unit_scale=True,
        desc=os.path.basename(output_path),
        leave=False
    )
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = len(chunk)
                f.write(chunk)
                progress.update(size)
    progress.close()

def export_project_data(url, api_key, project_id, output_dir):
    """Export all data from a Label Studio project.
    
    Args:
        url (str): Label Studio URL
        api_key (str): API key for authentication
        project_id (int): Project ID to export from
        output_dir (str): Directory to save exported data
        
    Returns:
        dict: Mapping between task IDs and their files
    """
    # Initialize client
    client = LabelStudioClient(url=url, api_key=api_key)
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Set up headers for authenticated requests
    headers = {'Authorization': f'Token {api_key}'}
    
    # Export annotations
    logger.info("Exporting annotations...")
    annotations = client.export_annotations(project_id, export_format='JSON')
    logger.info(f"Found {len(annotations)} tasks")
    
    # Create mapping for image-annotation pairs
    pairs_mapping = {}
    
    # Process each task
    logger.info("Processing tasks and downloading images...")
    for task in tqdm(annotations, desc="Processing tasks", unit="task"):
        try:
            task_id = str(task['id'])
            
            # Get image URL from task data
            image_path = task['data'].get('image')
            if not image_path:
                logger.warning(f"No image URL found in task {task_id}")
                continue
            
            # Combine the base URL with the image path
            if image_path.startswith('/'):
                image_url = urljoin(url, image_path)
            else:
                image_url = image_path
            
            # Generate output filenames
            original_filename = os.path.basename(urlparse(image_path).path)
            image_filename = f"task_{task_id}_{original_filename}"
            annotation_filename = f"task_{task_id}_annotation.json"
            
            image_output_path = os.path.join(images_dir, image_filename)
            annotation_output_path = os.path.join(annotations_dir, annotation_filename)
            
            # Add to mapping
            pairs_mapping[task_id] = {
                'image_file': image_filename,
                'annotation_file': annotation_filename,
                'original_filename': original_filename,
                'task_id': task_id
            }
            
            # Download image if it doesn't exist
            if not os.path.exists(image_output_path):
                download_file(image_url, image_output_path, headers=headers)
            
            # Save annotation data
            with open(annotation_output_path, 'w') as f:
                json.dump(task, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing task {task.get('id', 'unknown')}: {str(e)}")
            continue
    
    # Save the mapping file
    mapping_file = os.path.join(output_dir, "image_annotation_pairs.json")
    with open(mapping_file, 'w') as f:
        json.dump(pairs_mapping, f, indent=2, sort_keys=True)
    
    logger.info(f"Export complete! Data saved to: {os.path.abspath(output_dir)}")
    logger.info(f"- Images: {images_dir}")
    logger.info(f"- Annotations: {annotations_dir}")
    logger.info(f"- Mapping file: {mapping_file}")
    
    return pairs_mapping

def export_annotations(url, api_key, project_id):
    """Export and analyze annotations from a Label Studio project.
    
    Args:
        url (str): Label Studio URL
        api_key (str): API key for authentication
        project_id (int): Project ID to export from
        
    Returns:
        tuple: (all_annotations, valid_tasks) where:
            - all_annotations is the list of all task annotations
            - valid_tasks is the list of tasks with valid annotations
    """
    # Initialize client
    client = LabelStudioClient(url=url, api_key=api_key)
    
    # Export annotations
    logger.info("Exporting annotations...")
    annotations = client.export_annotations(project_id, export_format='JSON')
    
    logger.info(f"Total tasks: {len(annotations)}")
    
    # Count tasks with valid annotations (not cancelled and has results)
    valid_tasks = [
        task for task in annotations 
        if task.get('annotations') and 
        any(not ann.get('was_cancelled') and ann.get('result') 
            for ann in task['annotations'])
    ]
    
    logger.info(f"Tasks with valid annotations: {len(valid_tasks)}")
    
    if valid_tasks:
        # Show annotation types from first valid task
        example_task = valid_tasks[0]
        valid_annotation = next(
            ann for ann in example_task['annotations'] 
            if not ann.get('was_cancelled') and ann.get('result')
        )
        
        logger.info("\nAnnotation types:")
        for result in valid_annotation['result']:
            logger.info(f"\n- Type: {result['type']}")
            logger.info(f"  Value: {json.dumps(result['value'], indent=2)}")
    
    return annotations, valid_tasks 

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert (x_min, y_min, x_max, y_max) to YOLO format (x_center, y_center, width, height).
    All values are normalized to [0, 1].
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

def export_to_yolo(exported_data_dir, output_dir):
    """Convert exported Label Studio data to YOLO format.
    Assumes data has already been exported using export_project_data.
    
    Args:
        exported_data_dir (str): Path to the exported data directory containing:
            - annotations/: Directory with annotation JSON files
            - images/: Directory with image files
            - image_annotation_pairs.json: Mapping file
        output_dir (str): Directory to save YOLO format dataset
            
    Returns:
        int: Number of successfully processed annotations
    """
    # First load the exported data
    try:
        label_data, images_dir = load_label_studio_data(exported_data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise e
    
    # Create YOLO format output directories
    yolo_images_dir = os.path.join(output_dir, "images")
    yolo_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    logger.info(f"Converting {len(label_data)} annotations to YOLO format...")
    
    # Create dataset.yaml for YOLO
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images',  # relative to path
        'names': {
            0: 'cell'  # assuming single class 'cell'
        }
    }
    
    # Save dataset.yaml
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml_content = []
        yaml_content.append(f"path: {dataset_yaml['path']}")
        yaml_content.append(f"train: {dataset_yaml['train']}")
        yaml_content.append("names:")
        for idx, name in dataset_yaml['names'].items():
            yaml_content.append(f"  {idx}: {name}")
        f.write('\n'.join(yaml_content))
    
    successful_count = 0
    
    # Process each task with progress bar
    for task in tqdm(label_data, desc="Converting to YOLO format"):
        try:
            task_id = str(task['id'])
            image_filename = task['file_upload']
            
            # Get image path - image_filename already includes task_id prefix
            src_image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(src_image_path):
                logger.warning(f"Image not found: {src_image_path}")
                continue
            
            # Get image dimensions
            with Image.open(src_image_path) as img:
                img_width, img_height = img.size
            
            # Copy image to YOLO directory - keep the same filename
            dst_image_path = os.path.join(yolo_images_dir, image_filename)
            shutil.copy2(src_image_path, dst_image_path)
            
            # Create YOLO format label file - use same prefix as image
            label_filename = f"{os.path.splitext(image_filename)[0]}.txt"
            label_path = os.path.join(yolo_labels_dir, label_filename)
            
            yolo_annotations = []
            
            # Process each annotation
            for annotation in task.get('annotations', []):
                for result in annotation.get('result', []):
                    if result.get('type') == 'brushlabels':
                        # Decode mask
                        mask = decode_mask(result)
                        if mask is None:
                            continue
                        
                        # Get bounding box and convert to YOLO format
                        bbox = mask_to_bbox(mask)
                        yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
                        
                        # Format: <class> <x_center> <y_center> <width> <height>
                        yolo_line = f"0 {' '.join(f'{x:.6f}' for x in yolo_bbox)}"
                        yolo_annotations.append(yolo_line)
            
            # Save YOLO format annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
                
            successful_count += 1
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            continue
    
    logger.info(f"Export complete! Dataset saved to: {os.path.abspath(output_dir)}")
    logger.info(f"- Images: {yolo_images_dir}")
    logger.info(f"- Labels: {yolo_labels_dir}")
    logger.info(f"- Dataset config: {os.path.join(output_dir, 'dataset.yaml')}")
    
    return successful_count 