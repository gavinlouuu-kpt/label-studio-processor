#!/usr/bin/env python
# upload images to label studio according to the sqlite database that it is given
# the sqlite database contains the image paths and the group name of the image
# the group name is the name of the folder that the image is in
# label studio api key is db746855c4ae789b7ac4c06acde8c6f482da7c05
# label studio url is http://localhost:8080

import argparse
import logging
import os
import sqlite3
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from label_studio_processor.client import LabelStudioClient

logger = logging.getLogger(__name__)

def read_and_convert_image(image_path: str) -> Tuple[str, str]:
    """Read an image file, convert to PNG if it's a TIFF, and encode to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple[str, str]: (base64 encoded image, mime type)
    """
    try:
        ext = Path(image_path).suffix.lower()
        
        # If it's a TIFF, convert to PNG
        if ext in ['.tif', '.tiff']:
            with Image.open(image_path) as img:
                output = io.BytesIO()
                img.save(output, format='PNG')
                encoded = base64.b64encode(output.getvalue()).decode('utf-8')
                return encoded, 'image/png'
        else:
            # For other formats, just read and return
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                return encoded, mime_type
                
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {str(e)}")
        return None, None

def get_images_from_sqlite(db_path: str) -> List[Dict[str, Any]]:
    """Get image information from SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List[Dict[str, Any]]: List of image records with paths and group names
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Assuming the database has a table with image paths and group names
        cursor.execute("SELECT image_path, group_name FROM images")
        
        results = []
        for row in cursor.fetchall():
            image_path, group_name = row
            results.append({
                "image_path": image_path,
                "group_name": group_name
            })
            
        conn.close()
        return results
    
    except Exception as e:
        logger.error(f"Failed to read from SQLite database {db_path}: {str(e)}")
        raise

def upload_images_to_label_studio(
    db_path: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = "Image Classification",
    url: str = "http://localhost:8080",
    api_key: str = "db746855c4ae789b7ac4c06acde8c6f482da7c05"
) -> int:
    """Upload images from SQLite database to Label Studio.
    
    Args:
        db_path: Path to the SQLite database containing image information
        project_id: Optional ID of an existing project to upload to
        project_name: Name for the new project if project_id is not provided
        url: Label Studio instance URL
        api_key: API key for authentication
        
    Returns:
        int: Project ID
    """
    try:
        # Initialize client
        client = LabelStudioClient(url=url, api_key=api_key)
        
        # Create project if not provided
        if project_id is None:
            label_config = """
            <View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="cell_type" toName="image">
    <Label value="Original" background="#FF0000"/>
    <Label value="Bone" background="#00FF00"/>
    <Label value="Brain" background="#0000FF"/>
  </RectangleLabels>
  <Choices name="choice" toName="image" whenTagName="cell_type">
    <Choice value="Good Quality"/>
    <Choice value="Bad Quality"/>
  </Choices>
</View>
            """
            
            project_id = client.create_project(
                title=project_name,
                description=f"Images uploaded from SQLite database: {db_path}",
                label_config=label_config
            )
            logger.info(f"Created new Label Studio project with ID: {project_id}")
        else:
            # Verify the project exists
            client.get_project(project_id)
            logger.info(f"Using existing Label Studio project with ID: {project_id}")
        
        # Get images from database
        images = get_images_from_sqlite(db_path)
        logger.info(f"Found {len(images)} images in the database")
        
        # Prepare tasks for import
        tasks = []
        for img in images:
            image_path = img["image_path"]
            group_name = img["group_name"]
            
            # Check if the file exists
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            # Read, convert if needed, and encode the image
            base64_image, mime_type = read_and_convert_image(image_path)
            if not base64_image:
                continue
                
            task = {
                "data": {
                    "image": f"data:{mime_type};base64,{base64_image}",
                    "metadata": {
                        "file_path": image_path,
                        "group": group_name,
                        "filename": Path(image_path).name,
                        "converted": Path(image_path).suffix.lower() in ['.tif', '.tiff']
                    }
                }
            }
            tasks.append(task)
        
        # Import tasks in batches
        batch_size = 500
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            client.import_tasks(project_id, batch)
            logger.info(f"Imported {len(batch)} tasks to project {project_id}")
        
        logger.info(f"Successfully uploaded {len(tasks)} images to Label Studio project {project_id}")
        return project_id
        
    except Exception as e:
        logger.error(f"Failed to upload images to Label Studio: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload images from SQLite database to Label Studio")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--project-id", type=int, help="Existing Label Studio project ID")
    parser.add_argument("--project-name", default="Image Classification", help="Name for new project")
    parser.add_argument("--url", default="http://localhost:8080", help="Label Studio URL")
    parser.add_argument("--api-key", default="db746855c4ae789b7ac4c06acde8c6f482da7c05", help="Label Studio API key")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the upload process
    project_id = upload_images_to_label_studio(
        db_path=args.db,
        project_id=args.project_id,
        project_name=args.project_name,
        url=args.url,
        api_key=args.api_key
    )
    
    print(f"Images uploaded to project: {project_id}")
    print(f"Access the project at: {args.url}/projects/{project_id}")

if __name__ == "__main__":
    main()

