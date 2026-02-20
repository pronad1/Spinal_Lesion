# A Cascaded DERNet and YOLO11 Framework for Spinal Lesion Triage and Localization
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MICCAI 2026 Conference Submission**  
> Deep Learning-based Multi-Model Ensemble for Spinal Pathology Detection and Classification

## 📋 Overview

This repository contains the complete implementation of our proposed framework for automated detection and classification of spinal lesions from X-ray images using the VinDr-SpineXR dataset. Our approach combines state-of-the-art deep learning architectures to achieve superior performance across multiple evaluation metrics.

### Key Contributions

- **DERNet Ensemble**: Combines DenseNet-121, EfficientNetV2-S, and ResNet-50 via weighted soft-voting for high-sensitivity triage (91.03% AUROC)
- **YOLO11-L Detection**: CSP-Darknet backbone optimized for extreme class imbalance and small object detection (40.10% mAP@0.5)
- **Clinical Translation**: Deployment-ready web interface with automated outlier rejection for real-world clinical environments
- **Comprehensive Pipeline**: End-to-end framework from preprocessing (CLAHE) to localization with proven performance on VinDr-SpineXR benchmark

### Performance Highlights

| Task | Metric | Our Result (DERNet) | Baseline |
|------|--------|---------------------|----------|
| **Classification** | AUROC | 91.03% | 88.61% |
| | F1-Score | 83.09% | 81.06% |
| | Sensitivity | 84.91% | 83.07% |
| | Specificity | 81.68% | 79.32% |
| **Detection** | mAP@0.5 | 40.10% | 33.15% |

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
│   ├── train_densenet121.py          # DenseNet-121 training (86.93% AUROC)
│   ├── train_efficientnet.py         # EfficientNetV2-S training (89.44% AUROC)
│   ├── train_resnet50.py             # ResNet-50 training (88.88% AUROC)
│   └── ensemble_submission.py         # DERNet ensemble (91.03% AUROC)
│
├── detection/                         # Object detection models
│   ├── README.md                      # Detection details
│   └── train_yolo11l.py              # YOLO11-l training (40.10% mAP@0.5)
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
   - Dense connectivity optimizing feature reuse
   - Growth rate k=32, compression θ=0.5
   - Individual: **86.93% AUROC, 79.55% F1**

2. **EfficientNetV2-S** (21M params)
   - Compound scaling with Fused-MBConv blocks
   - Parameter efficiency optimization
   - Highest specificity: **91.12%** (70.80% Sensitivity)

3. **ResNet-50** (25.6M params)
   - Residual gradient flow optimization
   - Bottleneck architecture
   - Balanced: **88.88% AUROC, 82.72% Sensitivity**

**DERNet Strategy**: Weighted Soft-Voting with weights **[0.42, 0.32, 0.26]** prioritizing DenseNet for superior gradient flow
```
P_DERNet = 0.42·P_DenseNet + 0.32·P_EfficientNet + 0.26·P_ResNet
```
**DERNet Result**: **91.03% AUROC, 84.91% Sensitivity, 81.68% Specificity, 83.09% F1-Score**

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
- Optimizer: AdamW (lr=1e-4, weight_decay adjusted)
- Epochs: 55 (with mosaic augmentation cutoff)
- Batch size: 12 (RTX 3050 8GB optimized)
- Loss weights: λ_box=7.5, λ_cls=0.5, λ_dfl=1.5
- Augmentation: Mosaic (disabled after epoch 25), CLAHE preprocessing
- Final performance: **40.10% mAP@0.5**

For detailed mathematical formulations, see [`docs/methodology.md`](docs/methodology.md).

---

## 📈 Results

### Classification Results

| Model | AUROC (%) | F1-Score (%) | Sensitivity (%) | Specificity (%) |
|-------|-----------|--------------|-----------------|------------------|
| DenseNet-121 | 86.93 | 79.55 | 80.39 | 79.32 |
| EfficientNetV2-S | 89.44 | 79.34 | 70.80 | **91.12** |
| ResNet-50 | 88.88 | 80.15 | 82.72 | 78.13 |
| VinDr Baseline [9] | 88.61 | 81.06 | 83.07 | 79.32 |
| **DERNet (Ours)** | **91.03** | **83.09** | **84.91** | **81.68** |

### Detection Results (mAP@0.5)

| Class | YOLO11-L (Ours) | Sparse R-CNN | VinDr Baseline | EGCA-Net |
|-------|-----------------|--------------|----------------|----------|
| Disc Space Narrowing | 26.70% | 20.09% | 21.43% | 22.36% |
| Foraminal Stenosis | **41.40%** | 32.67% | 27.36% | 29.75% |
| Osteophytes | 40.60% | 48.16% | 34.78% | 36.73% |
| Spondylolisthesis | 54.80% | 45.32% | 41.29% | 44.69% |
| Surgical Implant | **74.10%** | 72.20% | 62.53% | 66.58% |
| Vertebral Collapse | **51.20%** | 49.30% | 43.39% | 50.41% |
| Other Lesions | 2.99% | 5.41% | 4.16% | 2.09% |
| **Overall mAP@0.5** | **40.10%** | 33.15% | 33.56% | 36.09% |

**Key Achievements**:
- **+11.1% improvement** over EGCA-Net (previous SOTA)
- **+41.40% AP** on Foraminal Stenosis (fine-grained detection)
- **+51.20% AP** on Vertebral Collapse (critical minority class)
- CSP-Darknet with C2PSA attention for small object detection

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
| Classification | DenseNet-121 | ~12 hours (60 epochs) | ~4 hours |
| Classification | EfficientNetV2-S | ~13 hours (60 epochs) | ~4.5 hours |
| Classification | ResNet-50 | ~12 hours (60 epochs) | ~4 hours |
| Detection | YOLO11-L | ~16 hours (55 epochs) | ~5.5 hours |
| **Total Pipeline** | All models | ~45 hours (single GPU) | ~15 hours (3 GPUs) |

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

**Prosenjit Mondol**  
Patuakhali Science & Technology University  
prosenjit1156@gmail.com

---

## 🙏 Acknowledgments

- VinDr Consortium for providing the VinDr-SpineXR dataset
- PyTorch and Ultralytics teams for their excellent deep learning frameworks
- PhysioNet for hosting and maintaining the dataset infrastructure

---

## 📧 Contact

For questions or collaborations:
- **Email**: prosenjit1156@gmail.com
- **GitHub Issues**: [Create an issue](https://github.com/pronad1/Spinal_Lesion/issues)

---

**Last Updated**: February 2026
