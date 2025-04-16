#!/usr/bin/env python
"""
This script creates a SQLite database with a table for storing image paths and group names. 
The script will be given a folder path and the script will recursively search for all image files in the folder.
The parent folder name of the image file will be the group name.
The script will create a table with the following columns:
- image_path: the path to the image file
- group_name: the name of the group that the image belongs to
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Common image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}

def find_image_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    Recursively find all image files in the given folder and extract their group names.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of tuples containing (image_path, group_name)
    """
    results = []
    folder_path = Path(folder_path).resolve()
    
    for root, _, files in os.walk(folder_path):
        # Get the parent folder name as the group name
        group_name = Path(root).name
        
        for file in files:
            # Check if the file is an image
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                file_path = Path(root) / file
                # Use absolute path for consistent referencing
                abs_path = str(file_path.resolve())
                results.append((abs_path, group_name))
                
    logger.info(f"Found {len(results)} image files in {folder_path}")
    return results

def create_sqlite_database(output_path: str, images: List[Tuple[str, str]]) -> str:
    """
    Create a SQLite database with a table for image paths and group names.
    
    Args:
        output_path: Path where the SQLite database will be saved
        images: List of tuples containing (image_path, group_name)
        
    Returns:
        str: The actual path where the database was created
    """
    try:
        # Convert to Path object for easier manipulation
        output_path = Path(output_path)
        
        # Check if the output path is a directory
        if output_path.is_dir():
            # If it's a directory, append a default database filename
            output_path = output_path / "images.db"
            logger.info(f"Output path is a directory, using file: {output_path}")
        
        # Ensure parent directory exists
        parent_dir = output_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {parent_dir}")
        
        # Create or connect to the database
        conn = sqlite3.connect(str(output_path))
        cursor = conn.cursor()
        
        # Create the table for storing image information
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            group_name TEXT NOT NULL
        )
        ''')
        
        # Clear existing data if the table already exists
        cursor.execute("DELETE FROM images")
        
        # Insert the image records
        cursor.executemany(
            "INSERT INTO images (image_path, group_name) VALUES (?, ?)", 
            images
        )
        
        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        
        logger.info(f"Created SQLite database at {output_path} with {len(images)} records")
        return str(output_path)
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        logger.error(f"Failed to create database at path: {output_path}")
        logger.error(f"Please check that the path is not a directory and you have write permissions.")
        raise
    except Exception as e:
        logger.error(f"Failed to create SQLite database: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Create a SQLite database for image files and their group names")
    parser.add_argument("--folder", required=True, help="Path to the folder containing image files")
    parser.add_argument("--output", required=True, help="Path where the SQLite database will be saved (file or directory)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Find image files
    images = find_image_files(args.folder)
    
    if not images:
        logger.warning(f"No image files found in {args.folder}")
        print(f"No image files found in {args.folder}")
        return
    
    # Create the SQLite database
    db_path = create_sqlite_database(args.output, images)
    
    print(f"Created SQLite database at {db_path} with {len(images)} image records")
    print(f"You can now use this database with the local_upload.py script:")
    print(f"python -m label_studio_processor.tools.local_upload --db '{db_path}'")

if __name__ == "__main__":
    main()