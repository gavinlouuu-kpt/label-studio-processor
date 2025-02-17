import logging
import os
from label_studio_processor.export import export_project_data
from label_studio_processor.client import AuthenticationError

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
    
    # Set up export directories
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    data_dir = os.path.join(workspace_root, "data")
    export_dir = os.path.join(data_dir, "example_exported_data")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Export project data
        export_project_data(
            url=BASE_URL,
            api_key=API_KEY,
            project_id=PROJECT_ID,
            output_dir=export_dir
        )
        
        logger.info(f"Data exported to: {export_dir}")
        
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {str(e)}")
    except ConnectionError as e:
        logger.error(f"Connection failed: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()