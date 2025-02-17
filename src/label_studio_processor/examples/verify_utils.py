from label_studio_processor.utils import decode_mask, mask_to_bbox
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def visualize_mask_and_bbox(image_path, mask, bbox, output_path):
    """Visualize the image with mask overlay and bounding box."""
    # Load image
    img = np.array(Image.open(image_path))
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Decoded Mask')
    ax2.axis('off')
    
    # Image with mask overlay and bbox
    ax3.imshow(img)
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
    
    # Paths
    exported_data_dir = "exported_data"
    images_dir = os.path.join(exported_data_dir, "images")
    annotations_dir = os.path.join(exported_data_dir, "annotations")
    output_dir = "verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mapping file
    mapping_file = os.path.join(exported_data_dir, "image_annotation_pairs.json")
    with open(mapping_file, 'r') as f:
        pairs_mapping = json.load(f)
    
    logger.info(f"Found {len(pairs_mapping)} image-annotation pairs")
    
    # Process each pair
    for task_id, pair_info in pairs_mapping.items():
        try:
            # Load annotation
            annotation_path = os.path.join(annotations_dir, pair_info['annotation_file'])
            with open(annotation_path, 'r') as f:
                task = json.load(f)
            
            # Get image path
            image_path = os.path.join(images_dir, pair_info['image_file'])
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Process each annotation result
            for annotation in task.get('annotations', []):
                for result in annotation.get('result', []):
                    if result.get('type') == 'brushlabels':
                        # Decode mask
                        mask = decode_mask(result)
                        if mask is None:
                            logger.warning(f"Failed to decode mask for task {task_id}")
                            continue
                        
                        # Get bounding box
                        bbox = mask_to_bbox(mask)
                        
                        # Visualize
                        output_path = os.path.join(output_dir, f"task_{task_id}_verification.png")
                        visualize_mask_and_bbox(image_path, mask, bbox, output_path)
                        logger.info(f"Generated verification for task {task_id}")
                        
                        # Only process the first brush annotation
                        break
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            continue
    
    logger.info(f"Verification complete! Results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main() 