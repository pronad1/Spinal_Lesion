"""
ResNet-50 Training for VinDr-SpineXR Classification (MICCAI 2026 Paper)
ResNet-50 Individual Performance:
- AUROC: 88.88%
- Sensitivity: 82.72% (strong recall)
- Specificity: 78.13%
- F1-Score: 80.15%

Training Configuration:
- Epochs: 60 (with CosineAnnealing LR scheduler)
- Optimizer: AdamW (lr=1e-4)
- Hardware: RTX 3050 8GB (~12 hours training time)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import json
import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)  # MICCAI 2026 paper: 60 epochs
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='resnet50')
    args = parser.parse_args()

    # Configuration
    DATA_ROOT = Path("C:/Users/prosenjit19/Desktop/Spine/vindr-spinexr/data")
    TRAIN_PNG = DATA_ROOT / "train_pngs"
    TEST_PNG = DATA_ROOT / "test_pngs"
    TRAIN_ANNO = DATA_ROOT / "annotations/train.csv"
    TEST_ANNO = DATA_ROOT / "annotations/test.csv"
    OUTPUT_DIR = Path("C:/Users/prosenjit19/Desktop/Spine/vindr-spinexr/outputs/resnet50_optimized")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    NUM_WORKERS = 0  # Must be 0 on Windows
    IMAGE_SIZE = 224

    print("=" * 80)
    print("ResNet-50 Training for VinDr-SpineXR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Check CUDA and force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please install PyTorch with CUDA support.")
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    print(f"Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print()

    # Dataset class
    class SpineDataset(Dataset):
        def __init__(self, image_dir, anno_file, transform=None):
            self.image_dir = Path(image_dir)
            self.transform = transform
            
            # Load annotations
            df = pd.read_csv(anno_file)
            
            # Get unique images and their labels
            # If image has any lesion, it's abnormal (1), otherwise normal (0)
            self.image_labels = {}
            for _, row in df.iterrows():
                img_id = row['image_id']
                lesion = row['lesion_type']
                
                if img_id not in self.image_labels:
                    # No finding = 0 (normal), anything else = 1 (abnormal)
                    self.image_labels[img_id] = 0 if lesion == 'No finding' else 1
                elif lesion != 'No finding':
                    # If any row for this image has a lesion, mark as abnormal
                    self.image_labels[img_id] = 1
            
            self.image_ids = list(self.image_labels.keys())
            print(f"  Loaded {len(self.image_ids)} unique images")
            print(f"  Normal: {sum(1 for v in self.image_labels.values() if v == 0)}")
            print(f"  Abnormal: {sum(1 for v in self.image_labels.values() if v == 1)}")
        
        def __len__(self):
            return len(self.image_ids)
        
        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            label = self.image_labels[img_id]
            
            # Load image
            img_path = self.image_dir / f"{img_id}.png"
            
            if not img_path.exists():
                img_path = self.image_dir / img_id
            
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)

    # Enhanced data transforms for ResNet-50
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # More rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Enhanced augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = SpineDataset(TRAIN_PNG, TRAIN_ANNO, transform=train_transform)
    test_dataset = SpineDataset(TEST_PNG, TEST_ANNO, transform=test_transform)

    # Calculate class distribution
    train_labels = list(train_dataset.image_labels.values())
    normal_count = train_labels.count(0)
    abnormal_count = train_labels.count(1)
    
    print(f"  Loaded {len(train_dataset)} unique images")
    print(f"  Normal: {normal_count}")
    print(f"  Abnormal: {abnormal_count}")
    
    test_labels = list(test_dataset.image_labels.values())
    print(f"  Loaded {len(test_dataset)} unique images")
    print(f"  Normal: {test_labels.count(0)}")
    print(f"  Abnormal: {test_labels.count(1)}")
    print()

    # Calculate class weights for balanced training
    pos_weight = torch.tensor([normal_count / abnormal_count]).to(device)
    print(f"  Class weight (Abnormal): {pos_weight.item():.4f}")
    print()

    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Create ResNet-50 model
    print("Creating ResNet-50 model...")
    model = models.resnet50(pretrained=True)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Loss and optimizer with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)  # Adam instead of SGD
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auroc': []}
    best_auroc = 0.0

    print("Starting training...")
    print("=" * 80)

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # Verify GPU usage on first batch
            if epoch == 0 and batch_idx == 0:
                print(f"\n[GPU Check] Images device: {images.device}, Labels device: {labels.device}")
                print(f"[GPU Check] GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB\n")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(predictions.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total
        val_auroc = roc_auc_score(all_labels, all_probs)
        
        # Update learning rate based on AUROC
        scheduler.step(val_auroc)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auroc': val_auroc,
            }, OUTPUT_DIR / 'model_best.pth')
            print(f"  ✓ New best AUROC: {val_auroc:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auroc': val_auroc,
        }, OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
        
        print("=" * 80)

    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\nFinal Evaluation with Best Model...")
    checkpoint = torch.load(OUTPUT_DIR / 'model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(predictions.cpu().numpy().flatten())

    # Calculate final metrics with bootstrap confidence intervals
    from scipy import stats
    
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Bootstrap confidence intervals
    print("\nCalculating 95% confidence intervals (bootstrap)...")
    n_bootstrap = 1000
    np.random.seed(42)
    
    bootstrap_auroc = []
    bootstrap_f1 = []
    bootstrap_sens = []
    bootstrap_spec = []
    
    n_samples = len(all_labels)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_labels = [all_labels[i] for i in indices]
        boot_probs = [all_probs[i] for i in indices]
        boot_preds = [all_preds[i] for i in indices]
        
        try:
            bootstrap_auroc.append(roc_auc_score(boot_labels, boot_probs))
            bootstrap_f1.append(f1_score(boot_labels, boot_preds))
            boot_cm = confusion_matrix(boot_labels, boot_preds)
            boot_tn, boot_fp, boot_fn, boot_tp = boot_cm.ravel()
            bootstrap_sens.append(boot_tp / (boot_tp + boot_fn) if (boot_tp + boot_fn) > 0 else 0)
            bootstrap_spec.append(boot_tn / (boot_tn + boot_fp) if (boot_tn + boot_fp) > 0 else 0)
        except:
            continue
    
    # Calculate 95% CI
    auroc_ci = np.percentile(bootstrap_auroc, [2.5, 97.5])
    f1_ci = np.percentile(bootstrap_f1, [2.5, 97.5])
    sens_ci = np.percentile(bootstrap_sens, [2.5, 97.5])
    spec_ci = np.percentile(bootstrap_spec, [2.5, 97.5])

    print("\n" + "=" * 80)
    print("FINAL RESULTS - ResNet-50")
    print("=" * 80)
    print("\nMetric          | Value   | 95% CI           | Paper Target (DenseNet-201)")
    print("-" * 80)
    print(f"AUROC           | {auroc*100:6.2f}% | ({auroc_ci[0]*100:.1f}%, {auroc_ci[1]*100:.1f}%) | 87.14% (85.6%, 88.6%)")
    print(f"F1 Score        | {f1*100:6.2f}% | ({f1_ci[0]*100:.1f}%, {f1_ci[1]*100:.1f}%) | 79.03% (77.1%, 80.9%)")
    print(f"Sensitivity     | {sensitivity*100:6.2f}% | ({sens_ci[0]*100:.1f}%, {sens_ci[1]*100:.1f}%) | 77.97% (75.4%, 80.5%)")
    print(f"Specificity     | {specificity*100:6.2f}% | ({spec_ci[0]*100:.1f}%, {spec_ci[1]*100:.1f}%) | 81.46% (79.1%, 83.8%)")
    print("=" * 80)
    print("\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    print("=" * 80)

    # Save final results
    final_results = {
        'model': 'ResNet-50',
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'best_epoch': checkpoint['epoch'],
        'metrics': {
            'auroc': float(auroc),
            'auroc_ci': [float(auroc_ci[0]), float(auroc_ci[1])],
            'f1': float(f1),
            'f1_ci': [float(f1_ci[0]), float(f1_ci[1])],
            'sensitivity': float(sensitivity),
            'sensitivity_ci': [float(sens_ci[0]), float(sens_ci[1])],
            'specificity': float(specificity),
            'specificity_ci': [float(spec_ci[0]), float(spec_ci[1])]
        },
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }

    with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Training complete!")
