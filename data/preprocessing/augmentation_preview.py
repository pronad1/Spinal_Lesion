"""
Augmentation Preview Tool

Visualizes what your data augmentations will look like.
Super useful for tuning augmentation parameters before training!

Usage:
    python augmentation_preview.py --image_path sample.png --num_samples 6
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


def get_augmentation_pipeline(image_size=384):
    """
    Create the same augmentation pipeline used in training.
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision transforms pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])


def preview_augmentations(image_path, num_samples=6, image_size=384, save_path=None):
    """
    Generate and display augmented versions of an image.
    
    Args:
        image_path: Path to input image
        num_samples: Number of augmented samples to generate
        image_size: Target image size
        save_path: Optional path to save the preview grid
    """
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return
    
    # Create augmentation pipeline
    augment = get_augmentation_pipeline(image_size)
    
    # Calculate grid dimensions
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Generate augmented samples
    print(f"🎨 Generating {num_samples} augmented samples...")
    
    for i in range(num_samples):
        # Apply augmentation
        augmented_tensor = augment(img)
        
        # Convert back to numpy for display
        augmented_np = augmented_tensor.permute(1, 2, 0).numpy()
        
        # Display
        axes[i].imshow(augmented_np)
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Augmentation Preview: {Path(image_path).name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved preview to: {save_path}")
    
    # Show
    plt.show()
    print("✅ Preview complete!")


def compare_original_vs_augmented(image_path, image_size=384, save_path=None):
    """
    Side-by-side comparison of original vs augmented image.
    
    Args:
        image_path: Path to input image
        image_size: Target image size
        save_path: Optional path to save comparison
    """
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return
    
    # Create pipelines
    basic_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    augment_transform = get_augmentation_pipeline(image_size)
    
    # Apply transforms
    original = basic_transform(img).permute(1, 2, 0).numpy()
    augmented = augment_transform(img).permute(1, 2, 0).numpy()
    
    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original (Resized)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(augmented)
    axes[1].set_title('Augmented', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f'Comparison: {Path(image_path).name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved comparison to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Preview data augmentations')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of augmented samples to generate (default: 6)')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Target image size (default: 384)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save preview image (optional)')
    parser.add_argument('--compare', action='store_true',
                        help='Show original vs augmented comparison instead')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return
    
    print(f"📷 Loading image: {args.image_path}")
    print(f"🎯 Target size: {args.image_size}×{args.image_size}\n")
    
    if args.compare:
        # Show comparison
        compare_original_vs_augmented(
            args.image_path,
            image_size=args.image_size,
            save_path=args.output
        )
    else:
        # Show augmentation grid
        preview_augmentations(
            args.image_path,
            num_samples=args.num_samples,
            image_size=args.image_size,
            save_path=args.output
        )


if __name__ == '__main__':
    main()
