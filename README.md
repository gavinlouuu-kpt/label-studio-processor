# Label Studio Interface

This package provides tools to interface with Label Studio for managing image annotations, particularly focused on cell segmentation tasks. It allows downloading original images, exporting annotations, and converting them into various formats including masks and YOLO-compatible bounding boxes.

## Features

- Download original images from Label Studio
- Export and process brush-based segmentation annotations
- Convert segmentation masks to bounding boxes
- Export data in YOLO format for object detection
- Visualization tools for verification
- Training data preparation utilities

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

Process the exported data and prepare it for training:

```bash
python -m label_studio_processor.examples.prepare_training_data
```

This script:
- Loads exported images and annotations
- Decodes brush-based segmentation masks
- Extracts or computes bounding boxes
- Generates visualizations for verification
- Provides dataset statistics

The prepared data includes:
- Original images
- Binary segmentation masks
- Bounding box coordinates
- Visualization of masks and boxes

### 3. Verify Annotations

Verify the exported annotations and generated masks/bounding boxes:

```bash
python -m label_studio_processor.examples.verify_utils
```

This generates visualizations showing:
- Original image
- Decoded segmentation mask
- Image with mask overlay and bounding box

### 4. Export in YOLO Format

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