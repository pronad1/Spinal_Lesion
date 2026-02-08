"""
Simple script to convert DICOM files to JPEG format
Can be used standalone without the Flask app running
"""
import pydicom
import cv2
import numpy as np
import os

def convert_dicom_to_jpeg(dicom_path, output_path=None):
    """
    Convert a DICOM file to JPEG format
    
    Args:
        dicom_path: Path to the DICOM file
        output_path: Optional output path for JPEG. If not provided, uses same name with .jpg extension
    
    Returns:
        Path to the saved JPEG file
    """
    try:
        # Read DICOM file
        print(f"Reading DICOM file: {dicom_path}")
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel data
        pixel_array = ds.pixel_array
        
        print(f"Image dimensions: {pixel_array.shape}")
        print(f"Pixel value range: {pixel_array.min()} - {pixel_array.max()}")
        
        # Normalize to 8-bit (0-255)
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE for better visualization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        pixel_array = clahe.apply(pixel_array)
        
        # Determine output path
        if output_path is None:
            output_path = os.path.splitext(dicom_path)[0] + '.jpg'
        
        # Save as JPEG
        print(f"Saving JPEG to: {output_path}")
        cv2.imwrite(output_path, pixel_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"✅ Successfully converted to JPEG!")
        print(f"Output file: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error converting DICOM to JPEG: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Convert the abnormal spine DICOM to JPEG
    dicom_file = r"Testing\abnormal.dicom"
    
    if os.path.exists(dicom_file):
        output_file = convert_dicom_to_jpeg(dicom_file, "abnormal_spine.jpg")
        
        if output_file:
            print(f"\n{'='*60}")
            print(f"Your abnormal spine image is ready!")
            print(f"Location: {os.path.abspath(output_file)}")
            print(f"{'='*60}")
    else:
        print(f"❌ File not found: {dicom_file}")
        print("\nAvailable test files:")
        if os.path.exists("Testing"):
            for file in os.listdir("Testing"):
                if file.endswith(('.dicom', '.dcm')):
                    print(f"  - Testing/{file}")
