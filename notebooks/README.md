# Jupyter Notebooks

This folder contains interactive Jupyter notebooks for dataset analysis and result visualization.

## 📓 Notebooks

### 1. Dataset Analysis (`01_dataset_analysis.ipynb`)

**Purpose**: Comprehensive exploration of the VinDr-SpineXR dataset

**Contents**:
- 📊 Class distribution analysis
- 📈 Imbalance ratio calculation (46.9:1)
- 📦 Bounding box size distribution
- 🔍 Object co-occurrence patterns
- 🖼️ Sample image visualization
- 📉 Statistical summaries

**Key Insights**:
- Extreme class imbalance: Osteophytes (82.1%) vs. Vertebral collapse (1.75%)
- Small object detection challenge: Mean area 8,812-9,745 px²
- Multi-label images: Average 2.3 pathologies per image

**Usage**:
```bash
jupyter notebook 01_dataset_analysis.ipynb
```

### 2. Visualization (`02_visualization.ipynb`)

**Purpose**: Visualize model predictions and performance metrics

**Contents**:
- 🎯 Detection results overlay
- 📊 Confusion matrices
- 📈 ROC curves (classification)
- 📉 Precision-Recall curves (detection)
- 🔬 Per-class performance analysis
- 🖼️ Qualitative results showcase

**Features**:
- Side-by-side comparison: Ground truth vs. Predictions
- False positive/negative analysis
- Confidence score distributions
- Ensemble prediction visualization

**Usage**:
```bash
jupyter notebook 02_visualization.ipynb
```

## 🚀 Quick Start

### Install Jupyter

```bash
pip install jupyter notebook matplotlib seaborn pandas numpy pillow pydicom
```

### Launch Notebook Server

```bash
# From the notebooks/ folder
cd notebooks
jupyter notebook

# Or from project root
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

### Running All Cells

In Jupyter:
1. Click **Kernel** → **Restart & Run All**
2. Or press `Ctrl+Enter` to run cells sequentially

## 📊 Sample Outputs

### Dataset Analysis

```
Class Distribution:
├── Osteophytes:           6,887 instances (82.1%)
├── Surgical implant:      5,409 instances (64.5%)
├── Disc space narrowing:  4,102 instances (48.9%)
├── Foraminal stenosis:    1,283 instances (15.3%)
├── Spondylolisthesis:       411 instances  (4.9%)
├── Other lesions:           446 instances  (2.9%)
└── Vertebral collapse:      268 instances  (1.75%)

Imbalance Ratio: 46.9:1
```

### Visualization Examples

#### Confusion Matrix (Classification)
```
                Predicted
              No Finding  Pathology
Ground Truth
No Finding        1,642        200
Pathology        1,093      5,454
```

#### Precision-Recall Curve (Detection)
```
Osteophytes:          AP = 42.3%
Surgical implant:     AP = 63.8%
Vertebral collapse:   AP = 31.2%
...
```

## 🛠️ Customization

### Modify Visualization Styles

```python
# In notebook cells
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
```

### Export Figures

```python
# Save high-resolution figures
plt.savefig('output/figure_name.png', dpi=300, bbox_inches='tight')
```

## 📦 Dependencies

All notebooks require:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import os

# For DICOM handling
import pydicom

# For metrics
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
```

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn pillow pydicom scikit-learn
```

## 🔍 Tips

1. **Run notebooks in order**: Start with `01_dataset_analysis.ipynb`
2. **Restart kernel if errors occur**: Kernel → Restart & Clear Output
3. **Adjust figure sizes** for better visibility in presentations
4. **Export key plots** for inclusion in papers/reports

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [Data README](../data/README.md) - Dataset setup
- [Methodology](../docs/methodology.md) - Detailed technical approach

---

**Need Help?** Check the main [README.md](../README.md) or open an issue on GitHub.
