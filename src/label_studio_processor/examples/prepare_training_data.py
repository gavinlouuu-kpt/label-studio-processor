import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from label_studio_processor.utils import prepare_training_data

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def visualize_sample(image, mask, bbox, output_path):
    """Visualize a single sample with its mask and bounding box."""
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

def main():
    logger = setup_logging()
    
    # Get workspace root (two levels up from the script location)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # Paths relative to workspace root
    exported_data_dir = os.path.join(workspace_root, "exported_data")
    annotations_dir = os.path.join(exported_data_dir, "annotations")
    images_dir = os.path.join(exported_data_dir, "images")
    output_dir = os.path.join(workspace_root, "training_data_visualization")
    
    logger.info(f"Using exported data from: {exported_data_dir}")
    logger.info(f"Using images from: {images_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mapping file
    mapping_path = os.path.join(exported_data_dir, "image_annotation_pairs.json")
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {mapping_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid mapping file: {mapping_path}")
        return
    
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
        logger.error("No valid annotation files found!")
        return
    
    # Prepare training data
    logger.info(f"Preparing training data from {len(label_data)} annotation files...")
    data = prepare_training_data(label_data, images_dir)
    
    # Get some statistics
    num_samples = len(data['images'])
    if num_samples == 0:
        logger.error("No valid samples found in the dataset!")
        return
    
    # Calculate average mask size and bbox size
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
    
    avg_mask_area = total_mask_area / num_samples
    avg_bbox_area = total_bbox_area / num_samples
    
    logger.info(f"Dataset statistics:")
    logger.info(f"- Number of samples: {num_samples}")
    logger.info(f"- Average mask area: {avg_mask_area:.2f} pixels")
    logger.info(f"- Average bbox area: {avg_bbox_area:.2f} pixels")
    
    # Visualize some samples
    logger.info("Generating visualizations...")
    for i, (task_id, image) in enumerate(list(data['images'].items())[:5]):
        mask = data['masks'][task_id]
        bbox = data['box_prompts'][task_id]
        
        output_path = os.path.join(output_dir, f"sample_{task_id}.png")
        visualize_sample(image, mask, bbox, output_path)
        
    logger.info(f"Visualizations saved to {output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 