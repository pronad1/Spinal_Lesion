# Data Preprocessing

Hey! 👋 This is where all the magic happens before we feed images to our models. Let's get your DICOM files ready for training!

## 📋 What's In Here

```
preprocessing/
├── README.md                    # You're reading it!
├── dicom_to_png.py             # Convert DICOM to PNG images
├── normalize_images.py          # Normalize and prepare images
├── data_split.py               # Split dataset into train/val/test
└── augmentation_preview.py      # Preview augmentations
```

## 🚀 Quick Start

### 1. Convert DICOM to PNG

First things first – let's convert those medical DICOM files to something more standard:

```bash
python dicom_to_png.py --input_dir ../sample_images --output_dir ../processed/images
```

**What this does**:
- Reads DICOM files
- Applies proper windowing for X-rays
- Converts to 8-bit PNG
- Preserves patient metadata in a CSV file

### 2. Normalize Images

Get your images ready for the models:

```bash
python normalize_images.py --input_dir ../processed/images --output_dir ../processed/normalized
```

**What this does**:
- Resizes to 384×384 (what our models expect)
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Normalizes pixel values to [0, 1]
- Saves in organized folders

### 3. Split Dataset

Let's organize your data for training:

```bash
python data_split.py --input_csv train.csv --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

**What this does**:
- Splits data into training (70%), validation (15%), and test (15%)
- Ensures balanced distribution of classes
- Creates separate CSV files for each split
- No data leakage – guaranteed!

### 4. Preview Augmentations

Want to see what your augmented images will look like?

```bash
python augmentation_preview.py --image_path sample.png --num_samples 6
```

**What this does**:
- Shows you 6 different augmented versions of your image
- Helps you tune augmentation parameters
- Saves a grid preview as PNG

## 🔧 Common Workflows

### Complete Pipeline (Start to Finish)

```bash
# Step 1: Convert DICOM to PNG
python dicom_to_png.py --input_dir ../sample_images --output_dir ../processed/images

# Step 2: Normalize images
python normalize_images.py --input_dir ../processed/images --output_dir ../processed/normalized

# Step 3: Split dataset
python data_split.py --input_csv metadata.csv --output_dir ../processed/splits

# Step 4: Check your augmentations
python augmentation_preview.py --image_path ../processed/normalized/sample.png
```

### Just Testing With Sample Data

```bash
# Quick test with sample images
python dicom_to_png.py --input_dir ../sample_images --output_dir ../test_output --limit 10
```

## 📊 Image Statistics

After preprocessing, you should see:
- **Input format**: DICOM (.dicom)
- **Output format**: PNG (.png)
- **Image dimensions**: 384×384 pixels
- **Color channels**: 3 (RGB, converted from grayscale)
- **Pixel range**: [0, 255] for PNG, [0, 1] for normalized

## 🎨 Windowing Parameters

For spinal X-rays, we use these windowing settings:
- **Window Center**: 40 HU (Hounsfield Units)
- **Window Width**: 400 HU

These values are optimized for bone visibility in spine imaging!

## 🔍 Quality Checks

Before training, verify your data:

```python
# Check image dimensions
from PIL import Image
img = Image.open('processed/normalized/image.png')
print(f"Size: {img.size}")  # Should be (384, 384)

# Check pixel value range
import numpy as np
arr = np.array(img)
print(f"Min: {arr.min()}, Max: {arr.max()}")  # Should be 0-255
```

## 💡 Tips & Tricks

**Out of disk space?**
- Process in batches using `--limit` flag
- Delete intermediate files after each step

**Images look weird?**
- Check windowing parameters in `dicom_to_png.py`
- Verify DICOM headers are correct
- Try previewing with `augmentation_preview.py`

**Unbalanced dataset?**
- Use stratified splitting in `data_split.py`
- Consider using class weights during training

**Need custom preprocessing?**
- All scripts are modular – edit them freely!
- Each script has helpful comments

## 📁 Expected Output Structure

After running all preprocessing:

```
data/
├── sample_images/              # Original DICOM files
├── processed/
│   ├── images/                 # Converted PNG files
│   ├── normalized/             # Normalized images (384×384)
│   ├── splits/
│   │   ├── train.csv          # Training set metadata
│   │   ├── val.csv            # Validation set metadata
│   │   └── test.csv           # Test set metadata
│   └── metadata.csv           # All images metadata
└── preprocessing/              # This folder!
```

## 🐛 Troubleshooting

**Error: "pydicom not found"**
```bash
pip install pydicom pillow opencv-python numpy pandas scikit-learn tqdm
```

**Error: "Invalid DICOM file"**
- Check if file is actually DICOM format
- Try opening with medical imaging software first
- Some files might be corrupted

**Images are too dark/bright**
- Adjust windowing in `dicom_to_png.py`
- Try using CLAHE enhancement
- Check your display settings!

## 🔗 Next Steps

Once preprocessing is done:
1. Head to [`../../classification/`](../../classification/) to train models
2. Or check [`../../detection/`](../../detection/) for object detection
3. Visualize your data in [`../../notebooks/`](../../notebooks/)

---

**Questions?** Check the main [README](../../README.md) or just ask! We're all learning here 🚀
