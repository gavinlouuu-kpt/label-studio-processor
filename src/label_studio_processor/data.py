"""Functions for preparing and processing Label Studio data."""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from .utils import prepare_training_data

logger = logging.getLogger(__name__)

def load_label_studio_data(exported_data_dir):
    """Load Label Studio exported data from directory.
    
    Args:
        exported_data_dir (str): Path to the exported data directory containing:
            - annotations/: Directory with annotation JSON files
            - images/: Directory with image files
            - image_annotation_pairs.json: Mapping file
            
    Returns:
        tuple: (label_data, images_dir) where:
            - label_data is a list of annotation dictionaries
            - images_dir is the path to the images directory
    """
    annotations_dir = os.path.join(exported_data_dir, "annotations")
    images_dir = os.path.join(exported_data_dir, "images")
    mapping_path = os.path.join(exported_data_dir, "image_annotation_pairs.json")
    
    # Verify directories exist
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load mapping file
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid mapping file: {mapping_path}")
    
    # Load all annotation files
    label_data = []
    for task_id, info in mapping.items():
        annotation_file = info['annotation_file']
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        try:
            with open(annotation_path, 'r') as f:
                task_data = json.load(f)
                # Add file_upload info from mapping
                task_data['file_upload'] = info['image_file']
                label_data.append(task_data)
        except FileNotFoundError:
            logger.warning(f"Annotation file not found: {annotation_path}")
            continue
        except json.JSONDecodeError:
            logger.warning(f"Invalid annotation file: {annotation_path}")
            continue
    
    if not label_data:
        raise ValueError("No valid annotation files found!")
        
    logger.info(f"Loaded {len(label_data)} annotation files")
    return label_data, images_dir

def get_dataset_statistics(data):
    """Calculate statistics for the prepared dataset.
    
    Args:
        data (dict): Dictionary containing:
            - images: Dict mapping image IDs to PIL Images
            - masks: Dict mapping image IDs to binary masks
            - box_prompts: Dict mapping image IDs to bounding boxes
            
    Returns:
        dict: Statistics including:
            - num_samples: Number of valid samples
            - avg_mask_area: Average mask area in pixels
            - avg_bbox_area: Average bounding box area in pixels
    """
    num_samples = len(data['images'])
    if num_samples == 0:
        return None
        
    total_mask_area = 0
    total_bbox_area = 0
    
    for task_id in data['images'].keys():
        mask = data['masks'][task_id]
        bbox = data['box_prompts'][task_id]
        
        # Mask area
        total_mask_area += np.sum(mask)
        
        # Bbox area
        x_min, y_min, x_max, y_max = bbox
        total_bbox_area += (x_max - x_min) * (y_max - y_min)
    
    return {
        'num_samples': num_samples,
        'avg_mask_area': total_mask_area / num_samples,
        'avg_bbox_area': total_bbox_area / num_samples
    }

def visualize_sample(image, mask, bbox, output_path):
    """Visualize a single sample with its mask and bounding box.
    
    Args:
        image (PIL.Image): Original image
        mask (numpy.ndarray): Binary mask
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        output_path (str): Path to save the visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Original image
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    # Image with overlays
    ax3.imshow(image_np)
    # Add semi-transparent mask overlay
    mask_overlay = np.zeros((*mask.shape, 4))  # RGBA
    mask_overlay[mask > 0] = [1, 0, 0, 0.3]  # Red with 0.3 alpha
    ax3.imshow(mask_overlay)
    
    # Add bounding box
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        fill=False, color='green', linewidth=2)
    ax3.add_patch(rect)
    ax3.set_title('Overlay with Bounding Box')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def prepare_and_visualize_data(exported_data_dir, output_dir=None, num_vis_samples=5):
    """Prepare training data and generate visualizations.
    
    Args:
        exported_data_dir (str): Path to the exported data directory
        output_dir (str, optional): Directory to save visualizations. If None,
            creates 'training_data_visualization' in the workspace root.
        num_vis_samples (int, optional): Number of samples to visualize. Defaults to 5.
        
    Returns:
        tuple: (prepared_data, statistics) where:
            - prepared_data is the dictionary of prepared training data
            - statistics is a dictionary of dataset statistics
    """
    # Load data
    label_data, images_dir = load_label_studio_data(exported_data_dir)
    
    # Prepare training data
    logger.info(f"Preparing training data from {len(label_data)} annotation files...")
    data = prepare_training_data(label_data, images_dir)
    
    # Calculate statistics
    statistics = get_dataset_statistics(data)
    if statistics:
        logger.info(f"Dataset statistics:")
        logger.info(f"- Number of samples: {statistics['num_samples']}")
        logger.info(f"- Average mask area: {statistics['avg_mask_area']:.2f} pixels")
        logger.info(f"- Average bbox area: {statistics['avg_bbox_area']:.2f} pixels")
    
    # Generate visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Generating visualizations...")
        
        for i, (task_id, image) in enumerate(list(data['images'].items())[:num_vis_samples]):
            mask = data['masks'][task_id]
            bbox = data['box_prompts'][task_id]
            
            output_path = os.path.join(output_dir, f"sample_{task_id}.png")
            visualize_sample(image, mask, bbox, output_path)
            
        logger.info(f"Visualizations saved to {output_dir}")
    
    return data, statistics 