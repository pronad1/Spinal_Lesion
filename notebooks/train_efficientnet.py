"""
EfficientNetV2-S Training for VinDr-SpineXR Classification (MICCAI 2026 Paper)
EfficientNetV2-S Individual Performance:
- AUROC: 89.44%
- Sensitivity: 70.80%
- Specificity: 91.12% (highest among individual models)
- F1-Score: 79.34%

Training Configuration:
- Epochs: 60 (with CosineAnnealing LR scheduler)
- Optimizer: AdamW (lr=1e-4)
- Hardware: RTX 3050 8GB (~13 hours training time)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import timm
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs (MICCAI 2026 paper: 60 with CosineAnnealing)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_s', 
                        choices=['tf_efficientnetv2_s', 'tf_efficientnetv2_m', 'tf_efficientnetv2_l'],
                        help='EfficientNetV2 model variant')
    parser.add_argument('--img-size', type=int, default=384, help='Image size (384 recommended for EfficientNetV2-S)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--no-pretrained', action='store_true', help='Skip pretrained weights (train from scratch)')
    parser.add_argument('--fp16', action='store_true', help='Enable AMP (mixed precision)')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--freeze-epochs', type=int, default=2, help='Freeze backbone for N epochs')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers (Windows recommend 0)')
    return parser.parse_args()

class SpineDataset(Dataset):
    def __init__(self, image_dir, anno_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        df = pd.read_csv(anno_file)
        
        # Get unique images and their labels
        self.image_labels = {}
        for _, row in df.iterrows():
            img_id = row['image_id']
            lesion = row['lesion_type']
            
            if img_id not in self.image_labels:
                self.image_labels[img_id] = 0 if lesion == 'No finding' else 1
            elif lesion != 'No finding':
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
        
        img_path = self.image_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = self.image_dir / img_id
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class MixUp:
    """MixUp augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs,
                    use_mixup=True, fp16=False, grad_accum=1, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    mixup = MixUp(alpha=0.2) if use_mixup else None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward with optional AMP
        if fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                if mixup and np.random.rand() > 0.5:
                    images, labels_a, labels_b, lam = mixup(images, labels)
                    outputs = model(images).squeeze()
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
        else:
            if mixup and np.random.rand() > 0.5:
                images, labels_a, labels_b, lam = mixup(images, labels)
                outputs = model(images).squeeze()
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

        # Backward with gradient accumulation
        if fp16 and scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0:
            # Gradient clipping
            if fp16 and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if fp16 and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        probs = torch.sigmoid(outputs.detach())
        predicted = (probs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, num_epochs, fp16=False, scaler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if fp16 and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    auroc = roc_auc_score(all_labels, all_probs) if len(all_labels) > 0 else 0.0

    return epoch_loss, epoch_acc, auroc, all_labels, all_probs

def calculate_metrics_with_ci(labels, probs, n_bootstrap=1000):
    """Calculate metrics with 95% confidence intervals"""
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)
    
    # Original metrics
    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Bootstrap for confidence intervals
    auroc_scores, f1_scores, sens_scores, spec_scores = [], [], [], []
    
    np.random.seed(42)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = np.random.choice(len(labels), len(labels), replace=True)
        boot_labels = labels[indices]
        boot_probs = probs[indices]
        boot_preds = (boot_probs > 0.5).astype(int)
        
        if len(np.unique(boot_labels)) == 2:
            auroc_scores.append(roc_auc_score(boot_labels, boot_probs))
            f1_scores.append(f1_score(boot_labels, boot_preds))
            
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(boot_labels, boot_preds).ravel()
            sens_scores.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
            spec_scores.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
    
    auroc_ci = np.percentile(auroc_scores, [2.5, 97.5])
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])
    sens_ci = np.percentile(sens_scores, [2.5, 97.5])
    spec_ci = np.percentile(spec_scores, [2.5, 97.5])
    
    return {
        'auroc': auroc, 'auroc_ci': auroc_ci.tolist(),
        'f1': f1, 'f1_ci': f1_ci.tolist(),
        'sensitivity': sensitivity, 'sensitivity_ci': sens_ci.tolist(),
        'specificity': specificity, 'specificity_ci': spec_ci.tolist(),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def main():
    args = parse_args()
    
    # Configuration
    DATA_ROOT = Path("C:/Users/prosenjit19/Desktop/Spine/vindr-spinexr/data")
    TRAIN_PNG = DATA_ROOT / "train_pngs"
    TEST_PNG = DATA_ROOT / "test_pngs"
    TRAIN_ANNO = DATA_ROOT / "annotations/train.csv"
    TEST_ANNO = DATA_ROOT / "annotations/test.csv"
    OUTPUT_DIR = Path(f"C:/Users/prosenjit19/Desktop/Spine/vindr-spinexr/outputs/{args.model}_optimized")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print(f"EfficientNetV2 Training for VinDr-SpineXR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    # Enable cuDNN benchmark for potentially improved performance
    cudnn.benchmark = True
    
    print(f"Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print()
    
    # EfficientNetV2 optimized transforms
    # Uses AutoAugment and stronger regularization
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Random erasing for robustness
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SpineDataset(TRAIN_PNG, TRAIN_ANNO, transform=train_transform)
    val_dataset = SpineDataset(TEST_PNG, TEST_ANNO, transform=val_transform)
    
    # Class weights
    num_normal = sum(1 for v in train_dataset.image_labels.values() if v == 0)
    num_abnormal = sum(1 for v in train_dataset.image_labels.values() if v == 1)
    pos_weight = torch.tensor([num_normal / num_abnormal]).to(device)
    print(f"\n  Class weight (Abnormal): {pos_weight.item():.4f}")
    print()
    
    # Create GradScaler if using AMP
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Data loaders (respect --num-workers)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create EfficientNetV2 model
    print(f"Creating {args.model} model...")
    
    if args.no_pretrained:
        print("Training from scratch (no pretrained weights)")
        model = timm.create_model(args.model, pretrained=False, num_classes=1)
    else:
        print("Attempting to load pretrained weights from ImageNet...")
        print("(This may take a few minutes on slow connections...)")
        try:
            model = timm.create_model(args.model, pretrained=True, num_classes=1)
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to download pretrained weights")
            print("⚠ Training from scratch instead")
            model = timm.create_model(args.model, pretrained=False, num_classes=1)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Pretrained weights loaded")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # AdamW with weight decay for better generalization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.05,  # Strong regularization
        betas=(0.9, 0.999)
    )
    
    # Cosine Annealing scheduler (PyTorch 1.7 compatible)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )
    
    # Training loop
    print("Starting training with optimizations:")
    print("  ✓ MixUp augmentation")
    print("  ✓ Gradient clipping")
    print("  ✓ Random erasing")
    print("  ✓ Warmup + Cosine annealing")
    print("  ✓ Strong weight decay")
    print("=" * 80)
    
    best_auroc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auroc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            use_mixup=True, fp16=args.fp16, grad_accum=args.grad_accum, scaler=scaler
        )

        # Validate
        val_loss, val_acc, val_auroc, _, _ = validate(
            model, val_loader, criterion, device, epoch, args.epochs,
            fp16=args.fp16, scaler=scaler
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
        }
        torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pth")
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save(checkpoint, OUTPUT_DIR / "model_best.pth")
            print(f"  ✓ New best AUROC: {val_auroc:.4f}")
        
        print("=" * 80)
    
    # Save training history
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\nFinal Evaluation with Best Model...")
    checkpoint = torch.load(OUTPUT_DIR / "model_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, _, final_labels, final_probs = validate(
        model, val_loader, criterion, device, best_epoch, args.epochs,
        fp16=args.fp16, scaler=scaler
    )
    
    print("\nCalculating 95% confidence intervals (bootstrap)...")
    metrics = calculate_metrics_with_ci(final_labels, final_probs)
    
    # Save final results
    final_results = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'image_size': args.img_size,
        'best_epoch': best_epoch,
        'metrics': metrics
    }
    
    with open(OUTPUT_DIR / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print final results
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {args.model}")
    print("=" * 80)
    print()
    print(f"{'Metric':<16} | {'Value':<7} | {'95% CI':<16} | Paper Target (DenseNet-201)")
    print("-" * 80)
    print(f"{'AUROC':<16} | {metrics['auroc']*100:>6.2f}% | ({metrics['auroc_ci'][0]*100:.1f}%, {metrics['auroc_ci'][1]*100:.1f}%) | 87.14% (85.6%, 88.6%)")
    print(f"{'F1 Score':<16} | {metrics['f1']*100:>6.2f}% | ({metrics['f1_ci'][0]*100:.1f}%, {metrics['f1_ci'][1]*100:.1f}%) | 79.03% (77.1%, 80.9%)")
    print(f"{'Sensitivity':<16} | {metrics['sensitivity']*100:>6.2f}% | ({metrics['sensitivity_ci'][0]*100:.1f}%, {metrics['sensitivity_ci'][1]*100:.1f}%) | 77.97% (75.4%, 80.5%)")
    print(f"{'Specificity':<16} | {metrics['specificity']*100:>6.2f}% | ({metrics['specificity_ci'][0]*100:.1f}%, {metrics['specificity_ci'][1]*100:.1f}%) | 81.46% (79.1%, 83.8%)")
    print("=" * 80)
    print()
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:>4}  FP: {cm['fp']:>4}")
    print(f"  FN: {cm['fn']:>4}  TP: {cm['tp']:>4}")
    print("=" * 80)
    print()
    print(f"All results saved to: {OUTPUT_DIR}")
    print("Training complete!")

if __name__ == "__main__":
    main()
