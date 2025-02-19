# Label Studio Interface

This package provides tools to interface with Label Studio for managing image annotations, particularly focused on cell segmentation tasks. It allows downloading original images, exporting annotations, and converting them into various formats including multiple instance masks and YOLO-compatible bounding boxes.

## Development Process

This project follows an Example-First Development (EFD) process, where we:

1. Start with concrete examples of what we want to achieve
2. Write example scripts that demonstrate the ideal workflow
3. Discover needed functionality through these examples
4. Extract and refactor common functions into modules
5. Iterate and improve based on real usage

### Example-First Development Flow

```
Example Script → Discover Needs → Extract Functions → Create Modules → Improve API
```

For instance, our YOLO export functionality evolved through:

1. **Start with Example**: Write a script to export data in YOLO format
```python
# First attempt in examples/export_yolo_format.py
def main():
    # Export annotations to YOLO format
    export_to_yolo(exported_dir, yolo_dir)
```

2. **Discover Needs**: Through the example, we discovered we needed:
- Loading Label Studio data
- Decoding mask annotations
- Converting masks to bounding boxes
- YOLO format conversion

3. **Extract Functions**: Created reusable functions:
```python
load_label_studio_data()  # For data loading
decode_mask()             # For mask handling
mask_to_bbox()           # For bbox conversion
bbox_to_yolo()           # For YOLO format
```

4. **Create Modules**: Organized into logical modules:
```
label_studio_processor/
├── data.py      # Data loading
├── utils.py     # Mask and bbox utilities
└── export.py    # Export functionality
```

5. **Improve API**: Refined based on usage:
```python
# Final version with better organization
export_project_data()  # Get data from Label Studio
export_to_yolo()      # Convert to YOLO format
```

### Benefits of This Approach

1. **Natural Discovery**
   - Real needs emerge from actual usage
   - Functions evolve based on practical requirements
   - API design is guided by concrete examples

2. **Clear Documentation**
   - Examples serve as living documentation
   - Usage patterns are clear from the start
   - Development process is transparent

3. **Iterative Improvement**
   - Start simple, add complexity as needed
   - Refactor based on real usage patterns
   - Easy to identify common functionality

4. **Practical Testing**
   - Examples serve as integration tests
   - Test cases come from real use cases
   - Easy to verify functionality

## Features

- Download original images from Label Studio
- Export and process brush-based segmentation annotations
- Support for multiple instance masks per image
- Convert segmentation masks to bounding boxes
- Export data in YOLO format for object detection
- Visualization tools for verification
- Training data preparation utilities
- Reusable data processing functions

## Using Exported Data for Training

The data exported by this package is designed to work seamlessly with our companion package `yolo-sam-training`. The workflow is:

1. **Export Data**:
   ```python
   python -m label_studio_processor.examples.prepare_training_data
   ```
   This creates a directory structure with:
   - Images in `training_data/images/`
   - Masks in `training_data/masks/`
   - YOLO boxes in `training_data/boxes/`
   - Dataset summary in `training_data/summary.json`

2. **Train SAM Model**:
   Install the companion package:
   ```bash
   pip install -e ../yolo-sam-training
   ```
   
   Then run training:
   ```python
   python -m yolo_sam_training.examples.sam_training_example
   ```

The exported data can also be used with other training frameworks that support YOLO format or instance segmentation masks.

## Installation

```bash
pip install -e .
```

## Usage

### 1. Download Images and Annotations

Export both images and their corresponding annotations:

```bash
python -m label_studio_processor.examples.export_images_and_annotations
```

This creates an organized export with:
- Images in `exported_data/images/`
- Annotations in `exported_data/annotations/`
- Mapping file `exported_data/image_annotation_pairs.json`

### 2. Prepare Training Data

You can prepare training data either using the example script:

```bash
python -m label_studio_processor.examples.prepare_training_data
```

Or by using the data module in your own code:

```python
from label_studio_processor.data import prepare_and_visualize_data

# Prepare data and generate visualizations
data, statistics = prepare_and_visualize_data(
    exported_data_dir="path/to/exported_data",
    output_dir="path/to/save/visualizations",
    num_vis_samples=5
)

# Access the prepared data
images = data['images']           # Dict of PIL Images
masks = data['masks']            # Dict of lists of binary masks
box_prompts = data['box_prompts'] # Dict of lists of bounding boxes
class_ids = data['class_ids']     # Dict of lists of class IDs

# Access statistics
num_samples = statistics['num_samples']
total_masks = statistics['total_masks']
avg_masks_per_image = statistics['avg_masks_per_image']
avg_mask_area = statistics['avg_mask_area']
avg_bbox_area = statistics['avg_bbox_area']
```

The data module provides several reusable functions:

```python
from label_studio_processor.data import (
    load_label_studio_data,
    prepare_training_data,
    get_dataset_statistics,
    visualize_sample
)

# Load exported data
label_data, images_dir = load_label_studio_data("path/to/exported_data")

# Prepare training data
data = prepare_training_data(label_data, images_dir)

# Get statistics
statistics = get_dataset_statistics(data)

# Visualize a single sample with multiple instances
visualize_sample(
    image=data['images']['task_id'],
    masks=data['masks']['task_id'],  # List of masks
    boxes=data['box_prompts']['task_id'],  # List of boxes
    class_ids=data['class_ids']['task_id'],  # List of class IDs
    output_path="sample_visualization.png"
)
```

### 3. Export in YOLO Format

Convert annotations to YOLO format for object detection:

```bash
python -m label_studio_processor.examples.export_yolo_format
```

Creates a YOLO-compatible dataset in `yolo_dataset/`:
- Images in `images/`
- YOLO format labels in `labels/`
- Dataset configuration in `dataset.yaml`

## Configuration

Set your Label Studio configuration in the example scripts:
```python
API_KEY = "your_api_key"
PROJECT_ID = your_project_id
BASE_URL = "http://your-label-studio-url"
```

## Data Processing

1. **Image Export**: Images are downloaded from Label Studio and saved locally

2. **Segmentation Masks**: 
   - Brush annotations are decoded using Label Studio SDK
   - Multiple masks per image are supported
   - Each mask corresponds to a separate instance
   - Masks are converted to binary format (0 or 1)
   - Masks are saved with index suffixes (e.g., task_id_0.png, task_id_1.png)

3. **Bounding Boxes**: 
   - Generated from segmentation masks using min/max coordinates
   - One box per mask instance
   - Or extracted from rectangle annotations if available
   - Maintains correspondence with masks

4. **YOLO Format**: 
   - Bounding boxes are normalized to [0,1] range
   - Each line in the label file corresponds to one instance
   - Format: `class_id x_center y_center width height`
   - Multiple lines per image for multiple instances

## Output Directory Structure

```
training_data/
├── images/
│   ├── task_161.png
│   └── task_162.png
├── masks/
│   ├── task_161_0.png  # First instance mask
│   ├── task_161_1.png  # Second instance mask
│   ├── task_162_0.png
│   └── task_162_1.png
├── boxes/
│   ├── task_161.txt    # Contains multiple boxes
│   └── task_162.txt
├── classes.txt         # Class name to ID mapping
└── summary.json       # Dataset statistics and metadata
```

## Directory Structure

```
label-studio-interface/
├── src/
│   └── label_studio_processor/
│       ├── utils.py              # Core utilities
│       ├── data.py              # Data processing functions
│       ├── client.py             # Label Studio client
│       └── examples/             # Example scripts
├── tests/                        # Test files
└── README.md                     # This file
```

## Requirements

- Python 3.7+
- label-studio-sdk
- numpy
- Pillow
- matplotlib (for visualization)
- tqdm (for progress bars)

## Notes

- Ensure Label Studio server is running and accessible
- Verify API key has necessary permissions
- For large datasets with multiple instances, exports may take longer
- Images must be uploaded to Label Studio server first
- Training data preparation requires exported data structure
- Memory usage increases with number of instances per image
- Consider batch processing for large datasets with many instances