import logging
import os
from label_studio_processor.export import export_project_data, export_to_yolo

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # Configuration
    API_KEY = "db746855c4ae789b7ac4c06acde8c6f482da7c05"
    PROJECT_ID = 3  # Mix beads project
    BASE_URL = "http://localhost:8080"
    
    # Set up directories
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    data_dir = os.path.join(workspace_root, "data")
    export_dir = os.path.join(data_dir, "example_exported_data")
    yolo_dir = os.path.join(data_dir, "example_yolo_dataset")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # First export the data from Label Studio
        logger.info("Step 1: Exporting data from Label Studio...")
        export_project_data(
            url=BASE_URL,
            api_key=API_KEY,
            project_id=PROJECT_ID,
            output_dir=export_dir
        )
        
        # Then convert to YOLO format
        logger.info("\nStep 2: Converting to YOLO format...")
        successful_count = export_to_yolo(
            exported_data_dir=export_dir,
            output_dir=yolo_dir
        )
        
        logger.info(f"Successfully converted {successful_count} annotations to YOLO format")
        logger.info(f"YOLO dataset saved to: {yolo_dir}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 