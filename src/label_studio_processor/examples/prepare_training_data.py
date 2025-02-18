import logging
import os
import json
import numpy as np
from PIL import Image
from label_studio_processor.utils import prepare_training_data, bbox_to_yolo

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_prepared_data(prepared_data, output_dir):
    """Save the prepared training data to disk.
    Handles multiple masks per image.
    
    Args:
        prepared_data (dict): Dictionary containing images, masks, box_prompts, class_ids and class_map
        output_dir (str): Directory to save the data
    """
    # Create subdirectories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    boxes_dir = os.path.join(output_dir, "boxes")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(boxes_dir, exist_ok=True)
    
    total_masks = 0
    
    # Save data for each task
    for task_id in prepared_data['images'].keys():
        # Save image
        image = prepared_data['images'][task_id]
        image.save(os.path.join(images_dir, f"{task_id}.png"))
        
        # Get image dimensions for YOLO format conversion
        img_width, img_height = image.size
        
        # Get all masks, boxes, and classes for this task
        task_masks = prepared_data['masks'][task_id]
        task_boxes = prepared_data['box_prompts'][task_id]
        task_classes = prepared_data['class_ids'][task_id]
        
        # Save each mask as a separate PNG with index
        for idx, mask in enumerate(task_masks):
            mask_filename = f"{task_id}_{idx}.png"
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(masks_dir, mask_filename))
        
        # Save all bounding boxes in YOLO format in a single file
        yolo_lines = []
        for bbox, class_id in zip(task_boxes, task_classes):
            yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
            yolo_lines.append(f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}")
        
        with open(os.path.join(boxes_dir, f"{task_id}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        total_masks += len(task_masks)
    
    # Save a summary file with class mapping and statistics
    summary = {
        'num_images': len(prepared_data['images']),
        'total_masks': total_masks,
        'task_ids': list(prepared_data['images'].keys()),
        'class_mapping': prepared_data['class_map']
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Save class mapping in YOLO format (classes.txt)
    # Sort by class ID to ensure consistent order
    class_items = sorted(prepared_data['class_map'].items(), key=lambda x: x[1])
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name, class_id in class_items:
            f.write(f"{class_id}: {class_name}\n")

def main():
    logger = setup_logging()
    
    # Set up directories
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    data_dir = os.path.join(workspace_root, "data")
    export_dir = os.path.join(data_dir, "example_exported_data")
    training_dir = os.path.join(data_dir, "example_training_data")
    
    # Get subdirectories from export
    images_dir = os.path.join(export_dir, "images")
    annotations_dir = os.path.join(export_dir, "annotations")
    
    # Create output directory
    os.makedirs(training_dir, exist_ok=True)
    
    # Load mapping file
    mapping_file = os.path.join(export_dir, "image_annotation_pairs.json")
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Mapping file not found at: {mapping_file}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in mapping file: {mapping_file}")
        return
    
    # Load tasks
    tasks = []
    for task_id, task_info in mapping_data.items():
        annotation_file = os.path.join(annotations_dir, task_info['annotation_file'])
        try:
            with open(annotation_file, 'r') as f:
                task = json.load(f)
                # Update the file_upload field with the correct image filename from mapping
                task['file_upload'] = task_info['image_file']
                tasks.append(task)
        except FileNotFoundError:
            logger.warning(f"Annotation file not found: {annotation_file}")
            continue
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in annotation file: {annotation_file}")
            continue
    
    # Prepare training data
    try:
        prepared_data = prepare_training_data(
            label_json=tasks,
            images_dir=images_dir
        )
        
        # Save the prepared data
        save_prepared_data(prepared_data, training_dir)
        
        logger.info(f"Successfully prepared {len(prepared_data['images'])} training samples")
        logger.info(f"Data saved to: {training_dir}")
        logger.info(f"- Images: {os.path.join(training_dir, 'images')}")
        logger.info(f"- Masks: {os.path.join(training_dir, 'masks')}")
        logger.info(f"- Boxes: {os.path.join(training_dir, 'boxes')}")
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 