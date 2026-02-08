"""
Image Normalization Script

Prepares images for deep learning by:
- Resizing to 384x384
- Applying CLAHE for contrast enhancement
- Normalizing pixel values

Usage:
    python normalize_images.py --input_dir ../processed/images --output_dir ../processed/normalized
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Makes features more visible without over-amplifying noise.
    
    Args:
        image: Input image (grayscale or RGB)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 2:
        # Grayscale image
        return clahe.apply(image)
    else:
        # RGB image - apply to each channel
        channels = cv2.split(image)
        enhanced_channels = [clahe.apply(ch) for ch in channels]
        return cv2.merge(enhanced_channels)


def normalize_image(image_path, output_path, target_size=(384, 384), apply_enhancement=True):
    """
    Normalize a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save normalized image
        target_size: Target dimensions (width, height)
        apply_enhancement: Whether to apply CLAHE
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not read: {image_path}")
            return False
        
        # Resize
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply CLAHE if requested
        if apply_enhancement:
            img_resized = apply_clahe(img_resized)
        
        # Save
        cv2.imwrite(str(output_path), img_resized)
        return True
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Normalize images for deep learning')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save normalized images')
    parser.add_argument('--size', type=int, default=384,
                        help='Target image size (default: 384)')
    parser.add_argument('--no_enhancement', action='store_true',
                        help='Disable CLAHE enhancement')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    input_dir = Path(args.input_dir)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    if not image_files:
        print(f"❌ No image files found in {args.input_dir}")
        return
    
    target_size = (args.size, args.size)
    enhancement_status = "disabled" if args.no_enhancement else "enabled"
    
    print(f"📁 Found {len(image_files)} images")
    print(f"🎯 Target size: {args.size}×{args.size}")
    print(f"✨ CLAHE enhancement: {enhancement_status}")
    print(f"💾 Output directory: {args.output_dir}\n")
    
    # Process images
    successful = 0
    for image_path in tqdm(image_files, desc="Normalizing images"):
        output_filename = image_path.stem + '.png'
        output_path = output_dir / output_filename
        
        if normalize_image(
            image_path, 
            output_path, 
            target_size,
            apply_enhancement=not args.no_enhancement
        ):
            successful += 1
    
    print(f"\n✅ Successfully processed {successful}/{len(image_files)} images")
    
    # Show sample statistics
    if successful > 0:
        sample_img = cv2.imread(str(output_dir / (image_files[0].stem + '.png')))
        print(f"\n📊 Sample image stats:")
        print(f"   Shape: {sample_img.shape}")
        print(f"   Min pixel value: {sample_img.min()}")
        print(f"   Max pixel value: {sample_img.max()}")
        print(f"   Mean pixel value: {sample_img.mean():.2f}")


if __name__ == '__main__':
    main()
