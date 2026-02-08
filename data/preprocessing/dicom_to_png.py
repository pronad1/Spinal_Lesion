"""
DICOM to PNG Converter for Spinal X-ray Images

This script converts DICOM medical images to PNG format with proper windowing
for bone/spine visualization. Perfect for getting your data ready for deep learning!

Usage:
    python dicom_to_png.py --input_dir ../sample_images --output_dir ../processed/images
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    print("❌ pydicom not found! Install it with: pip install pydicom")
    exit(1)


def apply_windowing(pixel_array, window_center=40, window_width=400):
    """
    Apply windowing to DICOM pixel data for optimal bone visualization.
    
    Args:
        pixel_array: Raw pixel data from DICOM
        window_center: Center of the window (HU) - default 40 for spine
        window_width: Width of the window (HU) - default 400 for bone
    
    Returns:
        Windowed image as uint8 array (0-255)
    """
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    
    # Apply windowing
    windowed = np.clip(pixel_array, img_min, img_max)
    
    # Normalize to 0-255
    windowed = ((windowed - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    
    return windowed


def dicom_to_png(dicom_path, output_path, window_center=40, window_width=400):
    """
    Convert a single DICOM file to PNG.
    
    Args:
        dicom_path: Path to input DICOM file
        output_path: Path to output PNG file
        window_center: Windowing center
        window_width: Windowing width
    
    Returns:
        Dictionary with metadata or None if failed
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        pixel_array = ds.pixel_array.astype(float)
        
        # Apply rescale if available
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Apply windowing
        windowed_img = apply_windowing(pixel_array, window_center, window_width)
        
        # Convert to RGB (models expect 3 channels)
        rgb_img = np.stack([windowed_img] * 3, axis=-1)
        
        # Save as PNG
        img = Image.fromarray(rgb_img)
        img.save(output_path)
        
        # Extract metadata
        metadata = {
            'original_filename': os.path.basename(dicom_path),
            'png_filename': os.path.basename(output_path),
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'rows': ds.Rows,
            'columns': ds.Columns,
        }
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error processing {dicom_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert DICOM images to PNG format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing DICOM files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save PNG files')
    parser.add_argument('--window_center', type=int, default=40,
                        help='Window center for bone visualization (default: 40)')
    parser.add_argument('--window_width', type=int, default=400,
                        help='Window width for bone visualization (default: 400)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files
    input_dir = Path(args.input_dir)
    dicom_files = list(input_dir.glob('*.dicom')) + list(input_dir.glob('*.dcm'))
    
    if not dicom_files:
        print(f"❌ No DICOM files found in {args.input_dir}")
        return
    
    # Apply limit if specified
    if args.limit:
        dicom_files = dicom_files[:args.limit]
    
    print(f"📁 Found {len(dicom_files)} DICOM files")
    print(f"🔧 Window settings: Center={args.window_center}, Width={args.window_width}")
    print(f"💾 Output directory: {args.output_dir}\n")
    
    # Process files
    metadata_list = []
    successful = 0
    
    for dicom_path in tqdm(dicom_files, desc="Converting DICOM to PNG"):
        # Create output filename
        output_filename = dicom_path.stem + '.png'
        output_path = output_dir / output_filename
        
        # Convert
        metadata = dicom_to_png(
            dicom_path, 
            output_path, 
            args.window_center, 
            args.window_width
        )
        
        if metadata:
            metadata_list.append(metadata)
            successful += 1
    
    # Save metadata
    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        metadata_path = output_dir / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\n✅ Successfully converted {successful}/{len(dicom_files)} files")
        print(f"📊 Metadata saved to: {metadata_path}")
    else:
        print(f"\n❌ No files were successfully converted")


if __name__ == '__main__':
    main()
