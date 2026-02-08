# Classification Models

This folder contains the implementation of three state-of-the-art CNN architectures for binary classification of spinal X-ray images (Pathology vs. No finding).

## 📋 Overview

Our ensemble classification framework combines three complementary deep learning architectures to achieve superior performance across multiple evaluation metrics.

### Models Included

| Model | Parameters | AUROC | Sensitivity | Specificity | F1-Score |
|-------|-----------|-------|-------------|-------------|----------|
| **DenseNet-121** | 8M | 90.25% | 83.32% | 82.34% | 82.46% |
| **EfficientNetV2-S** | 21M | 89.44% | 70.80% | 91.12% | 79.85% |
| **ResNet-50** | 25.6M | 88.88% | 82.72% | 78.13% | 80.42% |
| **Ensemble** | — | **90.25%** | **83.32%** | **82.34%** | **82.46%** |

## 🗂️ Files

```
classification/
├── README.md                    # This file
├── train_densenet121.py        # DenseNet-121 training script
├── train_efficientnet.py       # EfficientNetV2-S training script
├── train_resnet50.py           # ResNet-50 training script
└── ensemble_submission.py       # Ensemble inference and submission
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm
```

### Training Individual Models

#### 1. DenseNet-121

```bash
python train_densenet121.py
```

**Configuration**:
- Architecture: 4 dense blocks (6, 12, 24, 16 layers)
- Growth rate: k=32
- Compression factor: θ=0.5
- Input size: 384×384
- Batch size: 32
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)

**Expected Output**:
- Training time: ~4 hours (RTX 3050)
- Best checkpoint: `densenet121_best.pth`
- Final AUROC: ~90.25%

#### 2. EfficientNetV2-S

```bash
python train_efficientnet.py
```

**Configuration**:
- Architecture: Fused-MBConv blocks with progressive training
- Input size: 384×384 (progressive: 128→256→384)
- Batch size: 24
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=5e-5)
- Activation: SiLU (Swish)

**Expected Output**:
- Training time: ~5 hours (RTX 3050)
- Best checkpoint: `efficientnetv2_s_best.pth`
- Final AUROC: ~89.44%
- Highest specificity: 91.12%

#### 3. ResNet-50

```bash
python train_resnet50.py
```

**Configuration**:
- Architecture: 4 stages with bottleneck blocks
- Input size: 384×384
- Batch size: 32
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Activation: ReLU

**Expected Output**:
- Training time: ~6 hours (RTX 3050)
- Best checkpoint: `resnet50_best.pth`
- Final AUROC: ~88.88%

### Ensemble Prediction

After training all three models, generate ensemble predictions:

```bash
python ensemble_submission.py
```

**Ensemble Strategy**:
- Weighted average of probabilities
- Test-Time Augmentation (TTA): horizontal flip
- Optimal threshold search for F1-score maximization
- Final predictions: `ensemble_submission.csv`

## 🔬 Training Details

### Data Augmentation

All models use identical augmentation strategy:

```python
train_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Augmentations Applied**:
- Resize: 384×384 (consistent across all models)
- Horizontal flip: 50% probability
- Rotation: ±10 degrees
- Color jitter: ±20% brightness/contrast
- Normalization: ImageNet statistics

### Loss Functions

**Binary Cross-Entropy with Logits**:
```
Loss = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
```

where:
- y: ground truth label (0 or 1)
- ŷ: model logit output
- σ: sigmoid function

**With Class Weights** (optional for balanced training):
```python
pos_weight = torch.tensor([n_negative / n_positive])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Optimization

**AdamW Optimizer**:
- Decoupled weight decay (fixes Adam's L2 regularization issue)
- β₁ = 0.9, β₂ = 0.999
- ε = 1e-8
- Learning rate schedule: Cosine annealing

**Learning Rate Schedule**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

### Training Procedure

```python
# Pseudo-code for training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = torch.sigmoid(model(images))
            # Calculate metrics: AUROC, F1, Sensitivity, Specificity
    
    # Update learning rate
    scheduler.step()
    
    # Save best checkpoint based on AUROC
    if auroc > best_auroc:
        torch.save(model.state_dict(), 'best_checkpoint.pth')
```

## 📊 Evaluation Metrics

### 1. AUROC (Area Under ROC Curve)

```python
from sklearn.metrics import roc_auc_score
auroc = roc_auc_score(true_labels, predicted_probabilities)
```

**Interpretation**: Probability that a random positive sample ranks higher than a random negative sample.

### 2. Sensitivity (Recall/True Positive Rate)

```
Sensitivity = TP / (TP + FN)
```

**Clinical Importance**: Measures ability to detect all pathological cases (minimize false negatives).

### 3. Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

**Clinical Importance**: Measures ability to correctly identify healthy cases (minimize false positives).

### 4. F1-Score (Harmonic Mean of Precision and Recall)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

where:
```
Precision = TP / (TP + FP)
Recall = Sensitivity = TP / (TP + FN)
```

### Optimal Threshold Selection

Binary classification requires threshold τ to convert probabilities to labels:

```python
# Search for optimal threshold
thresholds = np.arange(0.35, 0.60, 0.001)
best_f1 = 0
best_threshold = 0.5

for tau in thresholds:
    predictions = (probabilities >= tau).astype(int)
    f1 = f1_score(true_labels, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = tau
```

## 🧪 Model Architecture Details

### DenseNet-121

**Key Features**:
- Dense connectivity: Each layer receives all previous feature maps
- Alleviates vanishing gradient problem
- Encourages feature reuse
- Fewer parameters than traditional CNNs

**Composite Function H(·)**:
```
H(x) = Conv3×3(ReLU(BatchNorm(x)))
```

**Dense Block Transition**:
```
x_ℓ = [x₀, x₁, ..., x_{ℓ-1}] ⊕ H_ℓ([x₀, x₁, ..., x_{ℓ-1}])
```

### EfficientNetV2-S

**Key Features**:
- Compound scaling: Balances depth, width, resolution
- Fused-MBConv blocks: Faster than MBConv
- Progressive training: Gradually increases image size
- Squeeze-and-Excitation (SE) attention

**Scaling Factors**:
- Depth: α^φ
- Width: β^φ
- Resolution: γ^φ
- Constraint: α·β²·γ² ≈ 2

### ResNet-50

**Key Features**:
- Residual connections: Enables training very deep networks
- Bottleneck design: 1×1, 3×3, 1×1 convolutions
- Batch normalization after each convolution
- Identity shortcuts for gradient flow

**Residual Block**:
```
y = F(x, {W_i}) + x
```

where F(·) represents the residual mapping.

## 💾 Model Checkpoints

After training, you'll have the following checkpoints:

```
checkpoints/
├── densenet121_best.pth          # Best DenseNet-121 (AUROC-based)
├── efficientnetv2_s_best.pth     # Best EfficientNetV2-S
├── resnet50_best.pth             # Best ResNet-50
└── training_logs/
    ├── densenet121_log.csv       # Training history
    ├── efficientnetv2_s_log.csv
    └── resnet50_log.csv
```

**Checkpoint Contents**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_auroc': best_auroc,
    'metrics': {
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score
    }
}
```

## 🔍 Troubleshooting

### Out of Memory (OOM) Error

**Solution 1**: Reduce batch size
```python
BATCH_SIZE = 16  # Instead of 32
```

**Solution 2**: Enable gradient accumulation
```python
accumulation_steps = 2
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Overfitting

**Symptoms**: High training accuracy, low validation accuracy

**Solutions**:
1. Increase data augmentation
2. Add dropout (p=0.5)
3. Increase weight decay
4. Use early stopping

### Slow Convergence

**Solutions**:
1. Increase learning rate (carefully)
2. Use learning rate warmup
3. Check data loading (use `num_workers=4`)
4. Enable mixed precision training

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📈 Expected Results Timeline

### DenseNet-121
- Epoch 10: AUROC ~85%
- Epoch 25: AUROC ~88%
- Epoch 40-50: AUROC converges to ~90%

### EfficientNetV2-S
- Epoch 15: AUROC ~86%
- Epoch 30: AUROC ~88.5%
- Epoch 45-50: AUROC converges to ~89.4%

### ResNet-50
- Epoch 10: AUROC ~84%
- Epoch 25: AUROC ~87%
- Epoch 40-50: AUROC converges to ~88.9%

## 🔗 References

1. Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
2. Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
3. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
4. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019

---

**For detailed methodology and mathematical formulations**, see [`../docs/methodology.md`](../docs/methodology.md)

**Need Help?** Check the main [README.md](../README.md) or open an issue on GitHub.
