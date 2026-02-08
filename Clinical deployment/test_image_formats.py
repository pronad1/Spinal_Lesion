"""
Test script to verify the app handles different image formats
"""
import requests
import os

# Test server URL
URL = "http://127.0.0.1:5000/upload"

# Test files (you can add your own test files)
test_cases = [
    # Format: (file_path, expected_behavior)
    ("test_spine.jpg", "Should process if spine image"),
    ("test_spine.png", "Should process if spine image"),
    ("test_chest.jpg", "Should reject - not a spine"),
    ("test.dcm", "Should process if spine DICOM"),
]

print("=" * 60)
print("Image Format Support Test")
print("=" * 60)

for file_path, expected in test_cases:
    if not os.path.exists(file_path):
        print(f"\n❌ {file_path}: File not found (skipped)")
        continue
    
    print(f"\n📁 Testing: {file_path}")
    print(f"   Expected: {expected}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/octet-stream')}
            response = requests.post(URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Status: SUCCESS")
            print(f"   Classification: {'Abnormal' if result['classification']['is_abnormal'] else 'Normal'}")
            if result.get('detection'):
                print(f"   Detections: {result['detection']['num_detections']}")
        else:
            result = response.json()
            print(f"   ⚠️ Status: REJECTED")
            print(f"   Message: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

print("\n" + "=" * 60)
print("Supported formats: .dcm, .dicom, .jpg, .jpeg, .png, .bmp, .tiff, .tif")
print("=" * 60)
