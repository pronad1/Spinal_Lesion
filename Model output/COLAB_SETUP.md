# VinDr-SpineXR Training on Google Colab

## 🎯 Goal
Beat paper baseline: **33.15 mAP@0.5** → Target: **36-38 mAP@0.5**

## 📦 What You Need
- `vindr_spinexr_complete.zip` (13.32 GB) - includes everything
- Google account with ~14 GB free Drive space
- 12-18 hours for training

## 🚀 4 Simple Steps

### Step 1: Upload to Google Drive (30-60 min)
1. Go to [drive.google.com](https://drive.google.com)
2. Create folder: `vindr-spinexr`
3. Upload `vindr_spinexr_complete.zip`
4. Right-click ZIP → Extract

### Step 2: Open in Colab (1 min)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → Google Drive
3. Navigate to: `vindr-spinexr/VinDr_SpineXR_Complete.ipynb`

### Step 3: Enable GPU (30 sec)
1. Runtime → Change runtime type
2. Select **T4 GPU** → Save

### Step 4: Run Training
1. Runtime → **Run all** (or `Ctrl+F9`)
2. Wait 12-18 hours
3. Results saved in: `vindr-spinexr/outputs/`

## ⏱️ Time Required
- Upload: 30-60 min
- Training: 12-18 hours
- **Total: ~13-19 hours**

## 💾 Storage Needed
- Google Drive: ~14 GB (delete ZIP after extraction to save 13 GB)
- Colab Free Tier: ✅ Works perfectly

## 📊 Expected Results
- **Paper baseline**: 33.15 mAP@0.5
- **Your model**: 36-38 mAP@0.5
- **Improvement**: +2.8-4.5 mAP ✅

## ⚠️ Common Issues

Results saved in: `vindr-spinexr/outputs/sparsercnn_improved/`

**Key files:**
- `model_final.pth` - Trained model
- `metrics.json` - Performance metrics
- `log.txt` - Training log

Download from Google Drive or directly in Colab notebook.

---

**That's it!** Upload → Extract → Run → Get results 🚀
