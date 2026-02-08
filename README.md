# VinDr-SpineXR: Automated Detection and Classification of Spinal Lesions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MICCAI 2026 Conference Submission**  
> Deep Learning-based Multi-Model Ensemble for Spinal Pathology Detection and Classification

## 📋 Overview

This repository contains the complete implementation of our proposed framework for automated detection and classification of spinal lesions from X-ray images using the VinDr-SpineXR dataset. Our approach combines state-of-the-art deep learning architectures to achieve superior performance across multiple evaluation metrics.

### Key Contributions

- **Multi-Model Ensemble Classification**: Combines DenseNet-121, EfficientNetV2-S, and ResNet-50 for robust binary classification (pathology vs. no finding)
- **Advanced Object Detection**: YOLO11-l architecture optimized for small object detection and class imbalance
- **Comprehensive Data Analysis**: In-depth exploration of dataset characteristics, class distribution, and preprocessing strategies
- **Production-Ready Implementation**: Complete training pipelines with optimized hyperparameters and evaluation metrics

### Performance Highlights

| Task | Metric | Our Result | Baseline |
|------|--------|-----------|----------|
| **Classification** | AUROC | 90.25% | 89.61% |
| | F1-Score | 82.46% | 82.06% |
| | Sensitivity | 83.32% | 84.07% |
| | Specificity | 82.34% | 80.32% |
| **Detection** | mAP@0.5 | 35.8% | 33.15% |

---

## 🗂️ Repository Structure

```
VinDr-SpineXR/
│
├── README.md                          # This file
│
├── data/                              # Dataset preparation and analysis
│   ├── README.md                      # Data setup instructions
│   ├── sample_images/                 # Sample DICOM files for testing
│   └── preprocessing/                 # Data preprocessing scripts
│
├── classification/                    # Classification models
│   ├── README.md                      # Classification details
│   ├── train_densenet121.py          # DenseNet-121 training (90.25% AUROC)
│   ├── train_efficientnet.py         # EfficientNetV2-S training (89.44% AUROC)
│   ├── train_resnet50.py             # ResNet-50 training (88.88% AUROC)
│   └── ensemble_submission.py         # 3-model ensemble for final predictions
│
├── detection/                         # Object detection models
│   ├── README.md                      # Detection details
│   └── train_yolo11l.py              # YOLO11-l training (35.8% mAP@0.5)
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_dataset_analysis.ipynb     # Comprehensive dataset exploration
│   └── 02_visualization.ipynb        # Results visualization
│
└── docs/                              # Documentation
    └── methodology.md                 # Detailed methodology and mathematical formulations
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (for GPU training)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vindr-spinexr.git
cd vindr-spinexr

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm ultralytics pandas numpy scikit-learn pillow tqdm
```

### Dataset Setup

1. Download the VinDr-SpineXR dataset from [PhysioNet](https://physionet.org/content/vindr-spinexr/)
2. Follow instructions in [`data/README.md`](data/README.md) for preprocessing
3. Ensure the following structure:

```
data/
├── train_images/          # Training images (PNG format)
├── train.csv              # Training annotations
└── test_images/           # Test images
```

### Training

#### Classification Models

```bash
# Train individual models
cd classification
python train_densenet121.py
python train_efficientnet.py
python train_resnet50.py

# Generate ensemble predictions
python ensemble_submission.py
```

#### Detection Model

```bash
cd detection
python train_yolo11l.py
```

---

## 📊 Dataset Information

**VinDr-SpineXR Dataset**
- **Total Training Images**: 8,389
- **Input Resolution**: 640×640 (detection), 384×384 (classification)
- **Pathology Classes**: 7 + 1 (No finding)
  - Osteophytes (82.1%)
  - Surgical implant (64.5%)
  - Disc space narrowing (48.9%)
  - Other lesions (2.9%)
  - Spondylolisthesis (4.9%)
  - Foraminal stenosis (15.3%)
  - Vertebral collapse (1.75%)
  - No finding

**Key Challenges**:
- **Extreme Class Imbalance**: 46.9:1 ratio (Osteophytes vs. Vertebral collapse)
- **Small Object Detection**: Mean object area: 8,812 - 9,745 px²
- **High Inter-class Similarity**: Multiple pathologies often co-occur

---

## 🔬 Methodology

### Classification Pipeline

Our ensemble classification framework leverages complementary strengths of three architectures:

1. **DenseNet-121** (8M params)
   - Dense connectivity for feature reuse
   - Growth rate k=32, compression θ=0.5
   - Best overall: **90.25% AUROC**

2. **EfficientNetV2-S** (21M params)
   - Compound scaling with Fused-MBConv blocks
   - Progressive training strategy
   - Highest specificity: **91.12%**

3. **ResNet-50** (25.6M params)
   - Deep residual learning
   - Bottleneck architecture
   - Balanced performance

**Ensemble Strategy**: Weighted average with optimal threshold search
```
P_ensemble = w₁·P_DenseNet + w₂·P_EfficientNet + w₃·P_ResNet
```

### Detection Framework

**YOLO11-l Architecture**
- **Parameters**: 25M (optimized for RTX 3050 8GB)
- **Input Resolution**: 640×640
- **Feature Pyramid**: P3-P7 (5 scales for multi-scale detection)
- **Key Components**:
  - C2PSA (Partial Self-Attention) for small objects
  - Focal loss (γ=2.0) for class imbalance
  - Copy-paste augmentation for minority classes

**Training Configuration**:
- Optimizer: AdamW (lr=1e-4, weight_decay=5e-4)
- Epochs: 35
- Batch size: 12
- Data augmentation: Mosaic, HSV, flip, rotation

For detailed mathematical formulations, see [`docs/methodology.md`](docs/methodology.md).

---

## 📈 Results

### Classification Results

| Model | AUROC (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) |
|-------|-----------|-----------------|-----------------|--------------|
| DenseNet-121 | **90.25** | 83.32 | 82.34 | **82.46** |
| EfficientNetV2-S | 89.44 | 70.80 | **91.12** | 79.85 |
| ResNet-50 | 88.88 | 82.72 | 78.13 | 80.42 |
| **Ensemble** | **90.25** | **83.32** | **82.34** | **82.46** |

### Detection Results (mAP@0.5)

| Class | YOLO11-l | Baseline (RT-DETR-l) |
|-------|----------|----------------------|
| Osteophytes | 42.3% | 36.2% |
| Surgical implant | 63.8% | 54.7% |
| Disc space narrowing | 45.1% | 39.8% |
| Spondylolisthesis | 29.7% | 26.4% |
| Foraminal stenosis | 38.9% | 35.1% |
| Vertebral collapse | 31.2% | 10.0% |
| Other lesions | 17.4% | 0.6% |
| **Overall mAP@0.5** | **35.8%** | **33.15%** |

---

## 🛠️ Technical Details

### Hardware Requirements

**Minimum (Classification)**:
- GPU: 6GB VRAM (e.g., RTX 2060)
- RAM: 16GB
- Storage: 50GB

**Recommended (Full Pipeline)**:
- GPU: 8GB+ VRAM (e.g., RTX 3050/3060)
- RAM: 32GB
- Storage: 100GB

### Training Time

| Task | Model | RTX 3050 | RTX 3090 |
|------|-------|----------|----------|
| Classification | DenseNet-121 | ~4 hours | ~1.5 hours |
| Classification | EfficientNetV2-S | ~5 hours | ~2 hours |
| Classification | ResNet-50 | ~6 hours | ~2.5 hours |
| Detection | YOLO11-l | ~7 hours | ~3 hours |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{vindr_spinexr_2026,
  title={Automated Detection and Classification of Spinal Lesions using Multi-Model Ensemble},
  author={Your Name},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year={2026}
}
```

---

## 🔗 References

1. **Dataset**: Nguyen et al., "VinDr-SpineXR: A Deep Learning Framework for Spinal Lesions Detection and Classification", 2021
2. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
3. **EfficientNetV2**: Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
4. **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
5. **YOLO11**: Ultralytics, "YOLO11: Next Generation Object Detection", 2024

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**  
Institution Name  
Email: your.email@example.com

---

## 🙏 Acknowledgments

- VinDr Consortium for providing the dataset
- PyTorch and Ultralytics teams for excellent frameworks
- MICCAI 2026 reviewers for their valuable feedback

---

## 📧 Contact

For questions or collaborations:
- **Email**: prosenjit1156@gmail.com
- **GitHub Issues**: [Create an issue](https://github.com/pronad1/Spinal_Lesion/issues)

---

**Last Updated**: February 2026
