# Quick Start Guide

## Running the Application Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

### 3. Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## Using the Application

1. **Upload a DICOM File**
   - Click on the upload area or drag & drop a `.dcm` or `.dicom` file
   - Maximum file size: 16MB

2. **View Results**
   - Classification: Normal or Abnormal with confidence scores
   - Detection: If abnormal, view detected lesions with bounding boxes
   - Metadata: Patient ID, study date, image dimensions

3. **Analyze Another Scan**
   - Click "Analyze Another Scan" button to upload a new file

## Testing the Application

### Sample Test with Notebook Data
If you have the VinDr-SpineXR dataset, you can test with those DICOM files:
```
/kaggle/input/vindr-spiner/.../train_images/*.dicom
```

### Expected Behavior

**For Normal Scans:**
- Status badge shows "✓ NORMAL" in green
- Lower ensemble score (< 0.449)
- No detection results displayed

**For Abnormal Scans:**
- Status badge shows "⚠️ ABNORMAL" in red
- Higher ensemble score (> 0.449)
- Detection results with bounding boxes
- List of detected lesions with locations

### Invalid File Handling

**Non-DICOM Files:**
- Error message: "Invalid file format"
- Prompt: "Please upload a valid DICOM (.dcm or .dicom) file"

**Corrupted DICOM:**
- Error message with specific PyDICOM error details

## Deployment Options

### Docker
```bash
# Build image
docker build -t spine-detection .

# Run container
docker run -p 5000:5000 spine-detection

# Access at http://localhost:5000
```

### Heroku
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Deploy
git push heroku main

# Open in browser
heroku open
```

**Note:** Heroku may require slug optimization due to model file sizes. Consider:
- Using S3/Azure Blob Storage for model files
- Implementing lazy loading
- Using Heroku's large dyno types

### Cloud Platforms

**AWS (EC2 + Elastic Beanstalk)**
```bash
eb init -p python-3.9 spine-detection
eb create spine-detection-env
eb open
```

**Google Cloud Platform (App Engine)**
```bash
gcloud app deploy
gcloud app browse
```

**Azure (App Service)**
```bash
az webapp up --name spine-detection --runtime "PYTHON:3.9"
```

## Troubleshooting

### Models Not Loading
- Verify model files exist in correct paths:
  - `ensemble output/densenet121_balanced/model_best.pth`
  - `ensemble output/resnet50_optimized/model_best.pth`
  - `ensemble output/tf_efficientnetv2_s_optimized/model_best.pth`
  - `detection output/yolo11/weights/best.pt`

### CUDA/GPU Issues
- The app automatically falls back to CPU if CUDA is unavailable
- For CPU-only: Models will still work but slower

### Memory Issues
- Large model files (~500MB total) require adequate RAM
- Minimum recommended: 4GB RAM
- For production: 8GB+ RAM

### Port Already in Use
```bash
# Change port in app.py (last line):
app.run(debug=True, host='0.0.0.0', port=8000)  # Use different port
```

## API Testing

### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "classification": 3,
    "detection": true
  }
}
```

### Upload Test (using curl)
```bash
curl -X POST -F "file=@path/to/sample.dcm" http://localhost:5000/upload
```

## Performance Notes

- **First Request**: Slower due to model loading (if not pre-loaded)
- **Subsequent Requests**: Fast (~2-5 seconds on GPU, ~10-20 seconds on CPU)
- **Concurrent Requests**: Single-threaded by default; use gunicorn for production

## Security Considerations

1. **File Validation**: Only DICOM files accepted
2. **Size Limits**: 16MB maximum to prevent DoS
3. **Temporary Storage**: Files auto-deleted after processing
4. **HTTPS**: Always use HTTPS in production
5. **Authentication**: Add user authentication for production deployments

## Next Steps

- Add user authentication system
- Implement result history/database
- Add PDF report generation
- Create REST API documentation
- Implement batch processing
- Add DICOM metadata validation
- Create admin dashboard for monitoring
