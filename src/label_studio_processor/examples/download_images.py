from label_studio_sdk import Client
from label_studio_processor.client import AuthenticationError
import os
import requests
import logging
import json
from urllib.parse import urlparse, urljoin

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_file(url, output_path, headers=None):
    """Download a file from URL to the specified path."""
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    logger = setup_logging()
    
    # Replace with your actual API key
    API_KEY = "db746855c4ae789b7ac4c06acde8c6f482da7c05"
    PROJECT_ID = 3
    BASE_URL = "http://localhost:8080"
    
    try:
        # Initialize client
        logger.info("Connecting to Label Studio...")
        ls = Client(
            url=BASE_URL,
            api_key=API_KEY
        )
        ls.check_connection()
        logger.info("Successfully connected to Label Studio")
        
        # Set up headers for authenticated requests
        headers = {
            'Authorization': f'Token {API_KEY}'
        }
        
        # Get the project
        project = ls.get_project(PROJECT_ID)
        
        # Get all tasks in the project
        logger.info("Fetching tasks...")
        tasks = project.get_tasks()
        logger.info(f"Found {len(tasks)} tasks")
        
        # Create output directories
        base_output_dir = "exported_data"
        images_dir = os.path.join(base_output_dir, "images")
        annotations_dir = os.path.join(base_output_dir, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create a mapping file to track image-annotation pairs
        pairs_mapping = {}
        
        # Download images and save annotations from each task
        for task in tasks:
            try:
                task_id = task['id']
                
                # Get image URL from task data
                image_path = task['data'].get('image')
                if not image_path:
                    logger.warning(f"No image URL found in task {task_id}")
                    continue
                
                # Combine the base URL with the image path
                if image_path.startswith('/'):
                    image_url = urljoin(BASE_URL, image_path)
                else:
                    image_url = image_path
                
                # Generate output filename from the original filename or task ID
                original_filename = os.path.basename(urlparse(image_path).path)
                image_filename = f"task_{task_id}_{original_filename}"
                image_output_path = os.path.join(images_dir, image_filename)
                
                # Save annotation data
                annotation_filename = f"task_{task_id}_annotation.json"
                annotation_output_path = os.path.join(annotations_dir, annotation_filename)
                
                # Add to mapping
                pairs_mapping[task_id] = {
                    'image_file': image_filename,
                    'annotation_file': annotation_filename,
                    'original_filename': original_filename
                }
                
                # Download image if it doesn't exist
                if not os.path.exists(image_output_path):
                    logger.info(f"Downloading image for task {task_id} to {image_filename}")
                    download_file(image_url, image_output_path, headers=headers)
                    logger.info(f"Successfully downloaded: {image_filename}")
                else:
                    logger.info(f"Image already exists: {image_output_path}")
                
                # Save annotation data
                with open(annotation_output_path, 'w') as f:
                    json.dump(task, f, indent=2)
                logger.info(f"Saved annotation data to: {annotation_filename}")
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Save the mapping file
        mapping_file = os.path.join(base_output_dir, "image_annotation_pairs.json")
        with open(mapping_file, 'w') as f:
            json.dump(pairs_mapping, f, indent=2)
        
        logger.info(f"Export complete! Data saved to: {os.path.abspath(base_output_dir)}")
        logger.info(f"- Images: {images_dir}")
        logger.info(f"- Annotations: {annotations_dir}")
        logger.info(f"- Mapping file: {mapping_file}")
            
    except AuthenticationError as e:
        logger.error(f"Authentication failed: {str(e)}")
    except ConnectionError as e:
        logger.error(f"Connection failed: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()