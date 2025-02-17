import logging
import os
import json
from label_studio_processor.utils import prepare_training_data

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # Set up directories
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    exported_data_dir = os.path.join(workspace_root, "exported_data")
    images_dir = os.path.join(exported_data_dir, "images")
    annotations_dir = os.path.join(exported_data_dir, "annotations")
    output_dir = os.path.join(workspace_root, "training_data_visualization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mapping file
    mapping_file = os.path.join(exported_data_dir, "image_annotation_pairs.json")
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
        
        logger.info(f"Successfully prepared {len(prepared_data['images'])} training samples")
        logger.info(f"Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 