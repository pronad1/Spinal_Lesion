from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import traceback
import warnings
from werkzeug.utils import secure_filename

# Memory optimization - set environment variables BEFORE importing heavy libraries
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib' if os.environ.get('RENDER') else os.path.join(os.getcwd(), '.matplotlib')
os.environ['YOLO_CONFIG_DIR'] = '/tmp'  # Fix Ultralytics config directory warning
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Disable matplotlib font manager logging and reduce font cache
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10

# Lazy imports - only load heavy libraries when needed
# This reduces startup memory usage significantly

def _get_torch():
    import torch
    return torch

def _get_numpy():
    import numpy as np
    return np

def _get_cv2():
    import cv2
    return cv2

def _get_transforms():
    from torchvision import transforms
    return transforms

def _get_timm():
    import timm
    return timm

def _get_yolo():
    from ultralytics import YOLO
    return YOLO

def _get_pydicom():
    import pydicom
    from pydicom.pixel_data_handlers import pylibjpeg_handler
    return pydicom

def _get_pil():
    from PIL import Image
    import io
    return Image, io

def _get_base64():
    import base64
    return base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

if os.environ.get('RENDER'):
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'dcm', 'dicom', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables - TRUE LAZY LOADING (on first request only)
classification_models = {}
detection_model = None
device = None  # Will be set when loading models
models_loaded = False  # Track if models have been loaded
model_load_lock = None  # Will be initialized as threading.Lock()

# Ensemble weights from final_results.json
ENSEMBLE_WEIGHTS = {
    'densenet121': 0.42,
    'efficientnet': 0.32,
    'resnet50': 0.26
}
CLASSIFICATION_THRESHOLD = 0.449


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load all trained models - only called on first request"""
    global classification_models, detection_model, device, models_loaded, model_load_lock
    
    # Thread safety for lazy loading
    import threading
    if model_load_lock is None:
        model_load_lock = threading.Lock()
    
    with model_load_lock:
        # Double-check locking pattern
        if models_loaded:
            return
        
        print("\n🔄 LOADING MODELS ON FIRST REQUEST (This may take 30-60 seconds)...")
        
        # Lazy load torch and set device
        torch = _get_torch()
        
        # Force CPU mode on low-memory environments to save memory
        # GPU memory allocation can be expensive even if not used
        device = torch.device('cpu')
        
        # Set inference mode globally - reduces memory usage
        torch.set_grad_enabled(False)
        
        timm = _get_timm()
        YOLO = _get_yolo()
        
        print("Loading models...")
        print(f"Device: {device} (CPU-only mode for memory optimization)")
        print(f"Current directory: {os.getcwd()}")
        
        # Load Classification Models
        try:
            # DenseNet121
            densenet = timm.create_model('densenet121', pretrained=False, num_classes=1)
            densenet_path = os.path.join('ensemble output', 'densenet121_balanced', 'model_best.pth')
            print(f"Looking for DenseNet at: {densenet_path}")
            if os.path.exists(densenet_path):
                checkpoint = torch.load(densenet_path, map_location=device, weights_only=False)
                # Handle checkpoint format (with 'model_state_dict' key or direct state dict)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                densenet.load_state_dict(state_dict)
                densenet.eval()
                # Keep on CPU
                classification_models['densenet121'] = densenet
                print("✓ DenseNet121 loaded")
                del checkpoint, state_dict  # Free memory immediately
            else:
                print(f"✗ DenseNet121 not found at {densenet_path}")
            
            # ResNet50
            resnet = timm.create_model('resnet50', pretrained=False, num_classes=1)
            resnet_path = os.path.join('ensemble output', 'resnet50_optimized', 'model_best.pth')
            print(f"Looking for ResNet50 at: {resnet_path}")
            if os.path.exists(resnet_path):
                checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                resnet.load_state_dict(state_dict)
                resnet.eval()
                classification_models['resnet50'] = resnet
                print("✓ ResNet50 loaded")
                del checkpoint, state_dict
            else:
                print(f"✗ ResNet50 not found at {resnet_path}")
            
            # EfficientNetV2-S
            efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=1)
            efficientnet_path = os.path.join('ensemble output', 'tf_efficientnetv2_s_optimized', 'model_best.pth')
            print(f"Looking for EfficientNet at: {efficientnet_path}")
            if os.path.exists(efficientnet_path):
                checkpoint = torch.load(efficientnet_path, map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                efficientnet.load_state_dict(state_dict)
                efficientnet.eval()
                classification_models['efficientnet'] = efficientnet
                print("✓ EfficientNetV2-S loaded")
                del checkpoint, state_dict
            else:
                print(f"✗ EfficientNet not found at {efficientnet_path}")
                
        except Exception as e:
            print(f"Warning: Error loading classification models: {e}")
            traceback.print_exc()
        
        # Load YOLO Detection Model
        try:
            yolo_path = os.path.join('detection output', 'yolo11', 'weights', 'best.pt')
            print(f"Looking for YOLO at: {yolo_path}")
            if os.path.exists(yolo_path):
                detection_model = YOLO(yolo_path)
                print("✓ YOLO11 detection model loaded")
            else:
                print(f"✗ YOLO not found at {yolo_path}")
        except Exception as e:
            print(f"Warning: Error loading detection model: {e}")
            traceback.print_exc()
        
        print(f"\n✅ Models loaded: {len(classification_models)} classification models, Detection: {detection_model is not None}")
        models_loaded = True
        print("🚀 Application ready to serve requests!\n")

def validate_dicom(file_path):
    """Validate if file is a proper DICOM file and check if it's a spine image"""
    try:
        pydicom = _get_pydicom()
        ds = pydicom.dcmread(file_path)
        
        # Check if it's a spine image by examining DICOM metadata
        body_part = str(getattr(ds, 'BodyPartExamined', '')).upper()
        study_description = str(getattr(ds, 'StudyDescription', '')).upper()
        series_description = str(getattr(ds, 'SeriesDescription', '')).upper()
        
        # Look for spine-related keywords
        spine_keywords = ['SPINE', 'VERTEBRA', 'LUMBAR', 'THORACIC', 'CERVICAL', 'SPINAL', 
                         'C-SPINE', 'L-SPINE', 'T-SPINE', 'DORSAL', 'SACRAL', 'COCCYX']
        
        # Look for non-spine keywords (things we definitely don't want)
        non_spine_keywords = ['CHEST', 'LUNG', 'BRAIN', 'HEAD', 'SKULL', 'ABDOMEN', 
                             'PELVIS', 'LEG', 'ARM', 'HAND', 'FOOT', 'KNEE', 'SHOULDER',
                             'ELBOW', 'WRIST', 'ANKLE', 'CARDIAC', 'HEART']
        
        # Check if any spine keyword is present
        all_text = f"{body_part} {study_description} {series_description}"
        
        # Check for non-spine keywords first (more restrictive)
        has_non_spine = any(keyword in all_text for keyword in non_spine_keywords)
        has_spine = any(keyword in all_text for keyword in spine_keywords)
        
        # If it explicitly has non-spine keywords, mark as not spine
        # If it has spine keywords, mark as spine
        # If no keywords at all, assume it might be spine (lenient for unlabeled files)
        if has_non_spine:
            is_spine = False
        elif has_spine:
            is_spine = True
        else:
            # No clear indicators - assume it's okay (lenient mode)
            is_spine = True
        
        return True, ds, is_spine, body_part, str(getattr(ds, 'Modality', 'Unknown'))
    except Exception as e:
        return False, str(e), False, '', ''


def check_if_spine_image(pixel_array):
    """Check if image appears to be a medical/spine X-ray or MRI"""
    # Note: YOLO model is trained to detect lesions/abnormalities, not spine presence
    # We'll use image characteristics to detect medical vs natural images
    
    try:
        np = _get_numpy()
        cv2 = _get_cv2()
        
        # Check if original image is color (RGB)
        is_color = len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3
        
        # Convert to grayscale for analysis
        if is_color:
            gray = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
            
            # Check 1: Determine if image is predominantly grayscale
            # Medical X-rays should have R≈G≈B for most pixels
            r_channel = pixel_array[:, :, 0].astype(np.float32)
            g_channel = pixel_array[:, :, 1].astype(np.float32)
            b_channel = pixel_array[:, :, 2].astype(np.float32)
            
            # Calculate how many pixels are grayscale (R≈G≈B within tolerance)
            rg_diff = np.abs(r_channel - g_channel)
            rb_diff = np.abs(r_channel - b_channel)
            gb_diff = np.abs(g_channel - b_channel)
            
            # A pixel is grayscale if all channel differences are < 10
            is_grayscale_pixel = (rg_diff < 10) & (rb_diff < 10) & (gb_diff < 10)
            grayscale_percentage = np.sum(is_grayscale_pixel) / is_grayscale_pixel.size
            
            # If image is predominantly grayscale (>85%), it's likely a medical X-ray
            if grayscale_percentage > 0.85:
                # Accept as medical image - it's grayscale
                pass  # Continue to other checks
            else:
                # Image has significant color content - check if it's a color photo
                avg_color_diff = (np.mean(rg_diff) + np.mean(rb_diff) + np.mean(gb_diff)) / 3.0
                
                # Calculate color saturation for non-grayscale images
                hsv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2HSV)
                saturation = hsv[:, :, 1]
                avg_saturation = np.mean(saturation)
                
                # If it has both color variation AND saturation, it's a color photo
                if avg_color_diff > 5.0 and avg_saturation > 10:
                    return False, "This appears to be a color photograph, not a medical X-ray"
        else:
            gray = pixel_array
        
        # Check 2: Resolution check - be lenient
        height, width = gray.shape
        if height < 50 or width < 50:
            return False, "Image resolution too low for medical analysis"
        
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Normalize histogram
        hist = hist / (height * width)
        
        
        max_bin_ratio = np.max(hist)
        if max_bin_ratio > 0.25:  # Too concentrated in one intensity
            return False, "Image histogram suggests a natural photo, not medical imaging"
        
        # Check 4: Edge density - medical images have specific edge patterns
        # Natural photos (like dogs) have many complex edges
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (height * width)
        
        if edge_ratio > 0.35:  # Too many edges for medical image
            return False, "Image has too many edges - appears to be a photograph"
        
        # Removed lower edge bound - some medical images may have few edges
        
        # Check 5: Contrast and intensity range - be lenient
        std_dev = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        intensity_range = max_val - min_val
        
        if std_dev < 10:  # Very low contrast - reduced from 15
            return False, "Image has insufficient contrast for medical analysis"
        
        if intensity_range < 50:  # Narrow intensity range - reduced from 80
            return False, "Image intensity range too narrow for medical imaging"
        
        # If all checks pass, accept as medical image
        return True, "Image appears to be a medical X-ray/MRI"
        
    except Exception as e:
        print(f"Warning: Image validation failed: {e}")
        traceback.print_exc()
        # On error, be lenient and accept
        return True, "Image accepted for analysis"


def preprocess_image(image_path, is_dicom=False):
    """Read and preprocess image (supports DICOM and regular image formats)"""
    if is_dicom:
        return preprocess_dicom(image_path)
    else:
        # Load regular image formats (jpg, png, etc.)
        try:
            Image, io = _get_pil()
            np = _get_numpy()
            cv2 = _get_cv2()
            
            # Read image using PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                if pil_image.mode == 'RGBA':
                    # Handle transparency
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            pixel_array = np.array(pil_image)
            
            # Check if image appears to be a medical image BEFORE converting to grayscale
            # This allows us to detect color photos
            is_medical, message = check_if_spine_image(pixel_array)
            print(f"Medical image check: {is_medical} - {message}")
            
            if not is_medical:
                raise ValueError(
                    "This appears to be a CHEST CT scan. "
                    "This system only analyzes spine X-rays and MRIs. "
                    "Please upload a spine-related DICOM file."
                )
            
            # Convert to grayscale for processing (to match DICOM pipeline)
            if len(pixel_array.shape) == 3:
                gray_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_array = pixel_array
            
            # Create mock DICOM metadata for consistency
            class MockDicom:
                PatientID = 'N/A'
                StudyDate = 'N/A'
                Modality = 'X-ray'
            
            return gray_array, MockDicom()
            
        except ValueError:
            # Re-raise spine detection errors
            raise
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")


def preprocess_dicom(dicom_path):
    """Read and preprocess DICOM image"""
    pydicom = _get_pydicom()
    np = _get_numpy()
    
    # Configure pydicom to use pylibjpeg for JPEG 2000 (better handling)
    from pydicom.pixel_data_handlers import pylibjpeg_handler
    pydicom.config.pixel_data_handlers = [pylibjpeg_handler]
    
    # Suppress JPEG 2000 bit depth warnings (these are harmless but noisy)
    warnings.filterwarnings('ignore', message='.*Bits Stored.*', category=UserWarning)
    
    # Validate DICOM and check if it's a spine image
    is_valid, result, is_spine, body_part, modality = validate_dicom(dicom_path)
    
    if not is_valid:
        raise ValueError(f"Invalid DICOM file: {result}")
    
    # Check if it's a spine image
    if not is_spine:
        if body_part and body_part != 'UNKNOWN' and body_part != '':
            raise ValueError(f"This appears to be a {body_part} {modality} scan. This system only analyzes spine X-rays and MRIs. Please upload a spine-related DICOM file.")
        else:
            raise ValueError("This DICOM file does not appear to be a spine X-ray or MRI. This system is specifically designed for spine analysis only.")
    
    ds = result
    pixel_array = ds.pixel_array
    
    # Handle MONOCHROME1 (inverted grayscale)
    # MONOCHROME1 means "0 is White", we want "0 is Black"
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array
    
    # Robust normalization to 0-255 range (fixes display issues on Linux/deployment)
    # Convert to float for precise calculations
    pixel_array = pixel_array.astype(np.float64)
    
    # Remove negative values and normalize
    pixel_array = np.maximum(pixel_array, 0)
    
    # Normalize to 0-255 range
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max > pixel_min:  # Avoid division by zero
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)
    
    # Convert to uint8 for consistency
    pixel_array = np.uint8(pixel_array)
    
    return pixel_array, ds


def classify_image(pixel_array):
    """Run ensemble classification"""
    torch = _get_torch()
    np = _get_numpy()
    Image, io = _get_pil()
    transforms = _get_transforms()
    
    # Transform for models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert grayscale to RGB
    if len(pixel_array.shape) == 2:
        rgb_image = np.stack([pixel_array] * 3, axis=-1)
    else:
        rgb_image = pixel_array
    
    pil_image = Image.fromarray(rgb_image)
    input_tensor = transform(pil_image).unsqueeze(0)  # Keep on CPU
    
    # Get predictions from each model
    predictions = {}
    with torch.no_grad():
        if 'densenet121' in classification_models:
            pred = torch.sigmoid(classification_models['densenet121'](input_tensor)).item()
            predictions['densenet121'] = pred
        
        if 'resnet50' in classification_models:
            pred = torch.sigmoid(classification_models['resnet50'](input_tensor)).item()
            predictions['resnet50'] = pred
        
        if 'efficientnet' in classification_models:
            pred = torch.sigmoid(classification_models['efficientnet'](input_tensor)).item()
            predictions['efficientnet'] = pred
    
    # Ensemble prediction
    ensemble_score = 0
    total_weight = 0
    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        if model_name in predictions:
            ensemble_score += predictions[model_name] * weight
            total_weight += weight
    
    if total_weight > 0:
        ensemble_score /= total_weight
    
    is_abnormal = ensemble_score > CLASSIFICATION_THRESHOLD
    
    return {
        'ensemble_score': float(ensemble_score),
        'is_abnormal': bool(is_abnormal),
        'confidence': float(max(ensemble_score, 1 - ensemble_score) * 100),
        'individual_predictions': predictions
    }


def detect_lesions(pixel_array, confidence_threshold=0.25):
    """Run YOLO detection with specified confidence threshold"""
    if detection_model is None:
        return None
    
    cv2 = _get_cv2()
    np = _get_numpy()
    
    # Convert to RGB for YOLO
    if len(pixel_array.shape) == 2:
        rgb_image = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = pixel_array
    
    # Run detection
    results = detection_model(rgb_image, conf=confidence_threshold)
    
    detections = []
    annotated_image = rgb_image.copy()
    
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name if available
                class_name = result.names[cls] if hasattr(result, 'names') else f"Class_{cls}"
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': class_name
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated_image, f"{class_name} {conf:.2f}", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 0), 2)
    
    # Convert annotated image to base64
    base64 = _get_base64()
    _, buffer = cv2.imencode('.png', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'num_detections': len(detections),
        'detections': detections,
        'annotated_image': img_base64,
        'confidence_threshold': confidence_threshold
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Lazy load models on first request
    if not models_loaded:
        load_models()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing file: {filename}")
        
        # Check if file extension is allowed
        if not allowed_file(filename):
            os.remove(filepath)
            return jsonify({
                'error': 'Invalid file format',
                'message': f'Please upload a valid image file. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}',
            }), 400
        
        # Determine if file is DICOM
        file_extension = filename.rsplit('.', 1)[1].lower()
        is_dicom = file_extension in {'dcm', 'dicom'}
        
        # Validate file can be read
        if is_dicom:
            try:
                pydicom = _get_pydicom()
                pydicom.dcmread(filepath)
            except Exception as e:
                os.remove(filepath)
                return jsonify({
                    'error': 'Invalid DICOM file',
                    'message': 'Unable to read DICOM file. Please ensure it is a valid DICOM format.',
                    'details': str(e)
                }), 400
        else:
            # Validate regular image file
            try:
                Image, io = _get_pil()
                Image.open(filepath).verify()
            except Exception as e:
                os.remove(filepath)
                return jsonify({
                    'error': 'Invalid image file',
                    'message': 'Unable to read image file. Please ensure it is a valid image format.',
                    'details': str(e)
                }), 400
        
        # Process image (DICOM or regular)
        try:
            pixel_array, dicom_data = preprocess_image(filepath, is_dicom=is_dicom)
        except ValueError as ve:
            # Handle non-spine images
            os.remove(filepath)
            error_message = str(ve)
            return jsonify({
                'error': 'Invalid spine image',
                'message': error_message
            }), 400
        
        # Run classification
        classification_result = classify_image(pixel_array)
        
        # Run detection if abnormal
        detection_result = None
        if classification_result['is_abnormal']:
            # First try with standard confidence (0.25)
            detection_result = detect_lesions(pixel_array, confidence_threshold=0.25)
            
            # If no detections found but classified as abnormal, try with lower confidence
            if detection_result and detection_result['num_detections'] == 0:
                print(f"No detections at 0.25 confidence, trying lower threshold for {filename}")
                # Try with lower confidence (0.15) to catch subtle abnormalities
                detection_result_low = detect_lesions(pixel_array, confidence_threshold=0.15)
                
                # If lower confidence found something, use it
                if detection_result_low and detection_result_low['num_detections'] > 0:
                    detection_result = detection_result_low
                    print(f"Found {detection_result_low['num_detections']} detections at 0.15 confidence")
                else:
                    # No detections even at low confidence
                    # This can happen: classification detects subtle patterns YOLO can't localize
                    print(f"Classification: Abnormal, but no visible lesions detected by YOLO")
        
        # Create preview image
        cv2 = _get_cv2()
        base64 = _get_base64()
        _, buffer = cv2.imencode('.png', pixel_array)
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get DICOM metadata
        metadata = {
            'patient_id': str(dicom_data.PatientID) if hasattr(dicom_data, 'PatientID') else 'N/A',
            'study_date': str(dicom_data.StudyDate) if hasattr(dicom_data, 'StudyDate') else 'N/A',
            'modality': str(dicom_data.Modality) if hasattr(dicom_data, 'Modality') else 'N/A',
            'image_size': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}"
        }
        
        result = {
            'success': True,
            'filename': filename,
            'metadata': metadata,
            'classification': classification_result,
            'detection': detection_result,
            'preview_image': preview_base64
        }
        
        # Clean up
        os.remove(filepath)
        
        print(f"Successfully processed {filename}")
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        # Log the full error
        print(f"Error processing upload: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/health')
def health():
    """Health check endpoint - returns immediately without loading models"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'model_count': {
            'classification': len(classification_models),
            'detection': detection_model is not None
        },
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'message': 'Models will load on first prediction request' if not models_loaded else 'Ready'
    })


@app.route('/convert-to-jpeg', methods=['POST'])
def convert_to_jpeg():
    """Convert DICOM image to JPEG format and return it"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Determine if file is DICOM
        file_extension = filename.rsplit('.', 1)[1].lower()
        is_dicom = file_extension in {'dcm', 'dicom'}
        
        if is_dicom:
            # Read DICOM and convert to JPEG
            pydicom = _get_pydicom()
            cv2 = _get_cv2()
            np = _get_numpy()
            
            ds = pydicom.dcmread(filepath)
            pixel_array = ds.pixel_data
            
            # Convert to numpy array
            if hasattr(pixel_array, 'tobytes'):
                pixel_array = np.frombuffer(pixel_array.tobytes(), dtype=np.uint16)
            else:
                pixel_array = np.frombuffer(pixel_array, dtype=np.uint16)
            
            # Reshape based on DICOM dimensions
            pixel_array = pixel_array.reshape((ds.Rows, ds.Columns))
            
            # Normalize to 8-bit (0-255)
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Apply CLAHE for better visualization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            pixel_array = clahe.apply(pixel_array)
            
            # Save as JPEG
            output_filename = filename.rsplit('.', 1)[0] + '.jpg'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, pixel_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Clean up DICOM file
            os.remove(filepath)
            
            # Return the JPEG file
            return send_file(
                output_path,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=output_filename
            )
        else:
            # Already an image file, just return it as JPEG
            Image, io = _get_pil()
            img = Image.open(filepath)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            output_filename = filename.rsplit('.', 1)[0] + '.jpg'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            img.save(output_path, 'JPEG', quality=95)
            
            # Clean up original file
            os.remove(filepath)
            
            return send_file(
                output_path,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=output_filename
            )
            
    except Exception as e:
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        
        print(f"Error converting to JPEG: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'error': 'Conversion failed',
            'message': str(e)
        }), 500


# DON'T load models at startup - save memory!
# Models will be loaded on first prediction request

if __name__ == '__main__':
    # Use environment variables for production
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode to avoid auto-reloader issues
    app.run(debug=False, host='0.0.0.0', port=port)

