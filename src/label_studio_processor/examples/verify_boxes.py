import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def plot_yolo_box(ax, box_line, img_width, img_height):
    """Plot a YOLO format box on the given axis.
    
    Args:
        ax: matplotlib axis
        box_line (str): YOLO format line "<class> <x_center> <y_center> <width> <height>"
        img_width (int): Image width
        img_height (int): Image height
    """
    # Parse YOLO format
    class_id, x_center, y_center, width, height = map(float, box_line.strip().split())
    
    # Convert normalized coordinates back to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate box corners
    x_min = x_center - width/2
    y_min = y_center - height/2
    
    # Create rectangle patch
    rect = patches.Rectangle(
        (x_min, y_min), width, height,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add class label
    ax.text(x_min, y_min-5, f'Class {int(class_id)}', color='r')

def verify_boxes(data_dir, num_samples=5):
    """Verify random samples of bounding boxes.
    
    Args:
        data_dir (str): Path to the data directory containing images and boxes
        num_samples (int): Number of random samples to verify
    """
    images_dir = os.path.join(data_dir, "images")
    boxes_dir = os.path.join(data_dir, "boxes")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # Select random samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(samples), figsize=(5*len(samples), 5))
    if len(samples) == 1:
        axes = [axes]
    
    # Plot each sample
    for ax, image_file in zip(axes, samples):
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        
        # Load corresponding box file
        box_file = os.path.join(boxes_dir, image_file.replace('.png', '.txt'))
        with open(box_file, 'r') as f:
            box_line = f.read().strip()
        
        # Plot image
        ax.imshow(image)
        
        # Plot box
        plot_yolo_box(ax, box_line, image.width, image.height)
        
        ax.set_title(f'Task {image_file.split(".")[0]}')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Create verification directory
    verify_dir = os.path.join(os.path.dirname(data_dir), "example_verification")
    os.makedirs(verify_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(verify_dir, "box_verification.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set paths
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    data_dir = os.path.join(workspace_root, "data", "example_training_data")
    
    # Verify boxes
    verify_boxes(data_dir) 