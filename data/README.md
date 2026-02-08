# Data Preparation Guide

This folder contains dataset preparation scripts and sample data for the VinDr-SpineXR project.

## 📁 Folder Structure

```
data/
├── README.md              # This file
├── sample_images/         # Sample DICOM files for testing
└── preprocessing/         # Data preprocessing and conversion scripts
```

## 📥 Dataset Download

### Step 1: Access VinDr-SpineXR Dataset

The VinDr-SpineXR dataset is available on PhysioNet:
- **URL**: https://physionet.org/content/vindr-spinexr/
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Size**: ~50GB (full dataset)

### Step 2: PhysioNet Account Setup

1. Create an account at https://physionet.org/
2. Complete CITI training for human subjects research
3. Request access to the VinDr-SpineXR dataset
4. Wait for approval (usually 1-2 business days)

### Step 3: Download Dataset

```bash
# Option 1: Using wget (recommended)
wget -r -N -c -np https://physionet.org/files/vindr-spinexr/1.0.0/

# Option 2: Using PhysioNet's download tool
pip install wfdb
wfdb-download vindr-spinexr
```

## 🔄 Data Preprocessing

### Format Conversion: DICOM to PNG

The original dataset contains DICOM files. Convert them to PNG format for faster training:

```python
import pydicom
from PIL import Image
import numpy as np
import os

def convert_dicom_to_png(dicom_path, output_path):
    """Convert DICOM file to PNG with proper windowing"""
    # Read DICOM
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(float)
    
    # Normalize to 0-255
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(img).save(output_path)

# Convert all training images
for dcm_file in os.listdir('path/to/dicom/train'):
    if dcm_file.endswith('.dicom'):
        dicom_path = os.path.join('path/to/dicom/train', dcm_file)
        png_name = dcm_file.replace('.dicom', '.png')
        png_path = os.path.join('train_images', png_name)
        convert_dicom_to_png(dicom_path, png_path)
```

### Annotation Format Conversion

Convert CSV annotations to COCO JSON format for object detection:

```python
import pandas as pd
import json

def csv_to_coco(csv_path, output_path):
    """Convert VinDr CSV annotations to COCO format"""
    df = pd.read_csv(csv_path)
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define categories
    classes = [
        "Osteophytes",
        "Surgical implant",
        "Disc space narrowing",
        "Spondylolisthesis",
        "Foraminal stenosis",
        "Vertebral collapse",
        "Other lesions"
    ]
    
    for idx, class_name in enumerate(classes):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "spine_lesion"
        })
    
    # Process annotations
    annotation_id = 0
    for image_id, group in df.groupby('image_id'):
        # Add image info
        coco_format["images"].append({
            "id": image_id,
            "file_name": f"{image_id}.png",
            "width": 640,  # Adjust based on your images
            "height": 640
        })
        
        # Add annotations for this image
        for _, row in group.iterrows():
            if row['lesion_type'] != 'No finding':
                x_min, y_min = row['x_min'], row['y_min']
                x_max, y_max = row['x_max'], row['y_max']
                width = x_max - x_min
                height = y_max - y_min
                
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": classes.index(row['lesion_type']),
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    # Save COCO JSON
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

# Convert annotations
csv_to_coco('train_annotations.csv', 'train_coco.json')
```

## 📊 Expected Directory Structure After Preprocessing

```
data/
├── sample_images/               # Sample DICOM files (included)
│   ├── 00a4089038fb4f7b926624bd31b3ca88.dicom
│   ├── 00a90a72fe7d73ba4793935f7c3c3ce9.dicom
│   └── 00aac5dcd9a01d49cdab7420f47a343d.dicom
│
├── train_images/                # PNG training images (after conversion)
│   ├── image_001.png
│   ├── image_002.png
│   └── ... (8,389 total)
│
├── test_images/                 # PNG test images
│   └── ... (test set)
│
├── annotations/                 # Processed annotations
│   ├── train.csv               # Classification labels
│   ├── train_coco.json         # Detection annotations (COCO format)
│   └── class_weights.json      # Class distribution info
│
└── preprocessing/               # Analysis notebooks
    └── (Jupyter notebooks for data exploration)
```

## 📈 Dataset Statistics

After downloading and preprocessing, you should have:

### Classification Task
- **Total Images**: 8,389 training images
- **Binary Labels**: Pathology (1) vs. No finding (0)
- **Class Distribution**:
  - Positive (Pathology): 6,547 images (78.0%)
  - Negative (No finding): 1,842 images (22.0%)

### Detection Task
- **Total Bounding Boxes**: ~35,000 annotations
- **Classes**: 7 pathology types
- **Class Distribution**:
  - Osteophytes: 82.1%
  - Surgical implant: 64.5%
  - Disc space narrowing: 48.9%
  - Foraminal stenosis: 15.3%
  - Spondylolisthesis: 4.9%
  - Other lesions: 2.9%
  - Vertebral collapse: 1.75%

### Image Properties
- **Format**: Grayscale X-ray (converted to 3-channel for models)
- **Original Size**: Variable (typically 2000×2000 to 3000×3000 pixels)
- **Preprocessed Size**: 
  - Classification: 384×384
  - Detection: 640×640

## 🔍 Data Exploration

Explore the dataset using provided Jupyter notebooks:

```bash
cd ../notebooks
jupyter notebook 01_dataset_analysis.ipynb
```

This notebook provides:
- Class distribution visualization
- Bounding box size analysis
- Image quality assessment
- Co-occurrence matrix of pathologies

## ⚠️ Important Notes

1. **PhysioNet Compliance**: 
   - Do not share or redistribute the dataset
   - Follow PhysioNet's data use agreement
   - Cite the original dataset paper in publications

2. **Preprocessing Time**:
   - DICOM to PNG conversion: ~2-3 hours for full dataset
   - Annotation processing: ~5-10 minutes

3. **Storage Requirements**:
   - Original DICOM: ~50GB
   - Converted PNG: ~35GB
   - Keep both for reproducibility

4. **Validation Split**:
   - The dataset provides official train/test split
   - For development, use 80/20 train/validation split
   - Stratify by class to maintain distribution

## 🔗 References

- **VinDr-SpineXR Paper**: [PhysioNet Link](https://physionet.org/content/vindr-spinexr/)
- **COCO Format**: [COCO Detection Format](https://cocodataset.org/#format-data)

## 💡 Tips

- Use `preprocessing/01_dataset_analysis.ipynb` to understand class imbalance
- Sample images in `sample_images/` can be used to test preprocessing pipeline
- For faster experimentation, create a subset (e.g., 1000 images) before full training

---

**Need Help?** Check the main [README.md](../README.md) or open an issue on GitHub.
