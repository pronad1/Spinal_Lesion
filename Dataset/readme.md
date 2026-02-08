# VinDr-SpineXR Dataset

## 📋 Quick Overview

**VinDr-SpineXR** is a large-scale public dataset for automated detection and classification of spinal abnormalities from X-ray images using deep learning.

- **Source**: Hospitals in Vietnam (2010-2020)
- **Total**: 5,000 studies (patients) → 10,468 images
- **Annotations**: Manual bounding boxes by expert radiologists (10+ years experience)

---

## 📊 Dataset Split

| Set | Studies | Images | Purpose |
|-----|---------|--------|---------|
| **Training** | 4,000 | 8,390 | Model training |
| **Test** | 1,000 | 2,078 | Model evaluation |

---

## 🏥 13 Lesion Classes

The dataset annotates **13 specific spine abnormalities**:

1. **Osteophytes** (bone spurs) - *Most common*
2. **Disc space narrowing**
3. **Surgical implant** (screws, rods)
4. **Foraminal stenosis** (nerve opening narrowing)
5. **Spondylolisthesis** (vertebra slipping)
6. **Vertebral collapse** (fracture/compression)
7. **Enthesophytes** (bony projections)
8. **Sclerotic lesion** (bone hardening)
9. **Subchondral sclerosis** (bone thickening below cartilage)
10. **Fracture**
11. **Foreign body**
12. **Ankylosis** (joint stiffness)
13. **Other lesions**

**Note**: Images with no abnormalities are labeled as **"No finding"**

---

## 📁 Data Structure

### Image Files
- **Format**: DICOM (`.dicom`)
- **Resolution**: ~2500 × 1600 pixels
- **Contains**: Medical metadata (age, sex) + high-quality pixel data

### Annotation File (`train.csv` / `annotations_train.csv`)

Each row = one bounding box annotation

| Column | Description | Example |
|--------|-------------|---------|
| `image_id` | Unique image identifier | `088ec4a2...` |
| `lesion_type` | Abnormality name | `Osteophytes` |
| `xmin` | Left coordinate (pixels) | `712.6` |
| `ymin` | Top coordinate (pixels) | `961.3` |
| `xmax` | Right coordinate (pixels) | `786.5` |
| `ymax` | Bottom coordinate (pixels) | `1011.7` |
| `rad_id` | Radiologist ID | `rad1` |

**Important**:
- **Normal images**: Single row with `lesion_type = "No finding"` (no coordinates)
- **Abnormal images**: Multiple rows possible (one per lesion)

---

## 🔬 Use Cases

The paper presents a **2-stage pipeline**:

1. **Stage 1: Classification** (Binary)
   - Is the spine healthy or abnormal?
   
2. **Stage 2: Detection** (Multi-class)
   - If abnormal, locate and classify the 7 key lesions

---

## 🚀 Getting Started

```python
import pandas as pd
import pydicom
from PIL import Image

# Load annotations
df = pd.read_csv('annotations_train.csv')

# Read DICOM image
dcm = pydicom.dcmread('00a4089038fb4f7b926624bd31b3ca88.dicom')
image_array = dcm.pixel_array

# Check for normal vs abnormal
normal_images = df[df['lesion_type'] == 'No finding']
abnormal_images = df[df['lesion_type'] != 'No finding']
```

---

## 📚 Reference

**Paper**: "VinDr-SpineXR: A deep learning framework for spinal lesions detection and classification from radiographs"

**Citation**: VinBigData Chest X-ray Abnormalities Detection

---

## 📂 Folder Structure

```
Dataset/
├── *.dicom                    # DICOM spine X-ray images
├── annotations_train.csv      # Bounding box annotations
├── readme.md                  # This file
└── png_output/               # Converted PNG images (if any)
```

---

**Last Updated**: December 2025