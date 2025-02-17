from label_studio_processor.utils import decode_mask, mask_to_bbox
import os
import json
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import shutil

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

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

def main():
    logger = setup_logging()
    
    # Paths
    exported_data_dir = "exported_data"
    images_dir = os.path.join(exported_data_dir, "images")
    annotations_dir = os.path.join(exported_data_dir, "annotations")
    
    # Create YOLO format output directories
    yolo_output_dir = "yolo_dataset"
    yolo_images_dir = os.path.join(yolo_output_dir, "images")
    yolo_labels_dir = os.path.join(yolo_output_dir, "labels")
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    # Load mapping file
    mapping_file = os.path.join(exported_data_dir, "image_annotation_pairs.json")
    with open(mapping_file, 'r') as f:
        pairs_mapping = json.load(f)
    
    logger.info(f"Found {len(pairs_mapping)} image-annotation pairs")
    
    # Create dataset.yaml for YOLO
    dataset_yaml = {
        'path': os.path.abspath(yolo_output_dir),
        'train': 'images',  # relative to path
        'names': {
            0: 'cell'  # assuming single class 'cell'
        }
    }
    
    # Save dataset.yaml
    with open(os.path.join(yolo_output_dir, 'dataset.yaml'), 'w') as f:
        yaml_content = []
        yaml_content.append(f"path: {dataset_yaml['path']}")
        yaml_content.append(f"train: {dataset_yaml['train']}")
        yaml_content.append("names:")
        for idx, name in dataset_yaml['names'].items():
            yaml_content.append(f"  {idx}: {name}")
        f.write('\n'.join(yaml_content))
    
    # Process each pair with progress bar
    for task_id, pair_info in tqdm(pairs_mapping.items(), desc="Processing annotations"):
        try:
            # Load annotation
            annotation_path = os.path.join(annotations_dir, pair_info['annotation_file'])
            with open(annotation_path, 'r') as f:
                task = json.load(f)
            
            # Get image path
            src_image_path = os.path.join(images_dir, pair_info['image_file'])
            if not os.path.exists(src_image_path):
                logger.warning(f"Image not found: {src_image_path}")
                continue
            
            # Get image dimensions
            with Image.open(src_image_path) as img:
                img_width, img_height = img.size
            
            # Copy image to YOLO directory
            dst_image_path = os.path.join(yolo_images_dir, pair_info['image_file'])
            shutil.copy2(src_image_path, dst_image_path)
            
            # Create YOLO format label file
            label_filename = os.path.splitext(pair_info['image_file'])[0] + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_filename)
            
            yolo_annotations = []
            
            # Process each annotation result
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
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            continue
    
    logger.info(f"Export complete! Dataset saved to: {os.path.abspath(yolo_output_dir)}")
    logger.info(f"- Images: {yolo_images_dir}")
    logger.info(f"- Labels: {yolo_labels_dir}")
    logger.info(f"- Dataset config: {os.path.join(yolo_output_dir, 'dataset.yaml')}")

if __name__ == "__main__":
    main() 