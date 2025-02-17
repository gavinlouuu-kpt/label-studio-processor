import logging
from label_studio_processor.export import export_annotations
from label_studio_processor.client import AuthenticationError
import json

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
    
    try:
        # Export and analyze annotations
        annotations, valid_tasks = export_annotations(
            url=BASE_URL,
            api_key=API_KEY,
            project_id=PROJECT_ID
        )
        
        print(f"\nExported annotations from project {PROJECT_ID}")
        print(f"Total tasks: {len(annotations)}")
        
        # Count tasks with valid annotations (not cancelled and has results)
        valid_tasks = [
            task for task in annotations 
            if task.get('annotations') and 
            any(not ann.get('was_cancelled') and ann.get('result') 
                for ann in task['annotations'])
        ]
        print(f"Tasks with valid annotations: {len(valid_tasks)}")
        
        if valid_tasks:
            # Show first task with valid annotations
            example_task = valid_tasks[0]
            print("\nExample annotated task:")
            print(json.dumps(example_task, indent=2))
            
            # Show valid annotation types
            valid_annotation = next(
                ann for ann in example_task['annotations'] 
                if not ann.get('was_cancelled') and ann.get('result')
            )
            
            print("\nAnnotation types:")
            for result in valid_annotation['result']:
                print(f"\n- Type: {result['type']}")
                print(f"  Value: {json.dumps(result['value'], indent=2)}")
        else:
            print("\nNo tasks with valid annotations found")
            
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