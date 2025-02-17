# Label Studio Interface

This package provides tools to interface with Label Studio for managing image annotations, particularly focused on cell segmentation tasks. It allows downloading original images, exporting annotations, and converting them into various formats including masks and YOLO-compatible bounding boxes.

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
- Convert segmentation masks to bounding boxes
- Export data in YOLO format for object detection
- Visualization tools for verification
- Training data preparation utilities
- Reusable data processing functions

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
images = data['images']          # Dict of PIL Images
masks = data['masks']           # Dict of binary masks
box_prompts = data['box_prompts'] # Dict of bounding boxes

# Access statistics
num_samples = statistics['num_samples']
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

# Visualize a single sample
visualize_sample(
    image=data['images']['task_id'],
    mask=data['masks']['task_id'],
    bbox=data['box_prompts']['task_id'],
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
   - Masks are converted to binary format (0 or 1)
3. **Bounding Boxes**: 
   - Generated from segmentation masks using min/max coordinates
   - Or extracted from rectangle annotations if available
4. **YOLO Format**: Bounding boxes are normalized to [0,1] range

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
- For large datasets, exports may take some time
- Images must be uploaded to Label Studio server first
- Training data preparation requires exported data structure