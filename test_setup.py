"""
Test Script - Verify SignSync AI Setup
Checks all dependencies and files before running inference
"""

import sys

def check_imports():
    """Check if all required libraries are installed"""
    print("📦 Checking Python packages...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'mediapipe': 'mediapipe',
        'torch': 'torch (PyTorch)',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {install_name}")
        except ImportError:
            print(f"   ❌ {install_name} - NOT INSTALLED")
            missing.append(install_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All packages installed!\n")
    return True

def check_files():
    """Check if model files exist"""
    import os
    
    print("📁 Checking for model files...")
    
    files = {
        'sign_language_model_v2.pth': 'Trained LSTM model (PyTorch)',
        'label_map.npy': 'Class labels',
        'real_time_inference.py': 'Inference script',
        'process_sentence_videos.py': 'Data processing script',
        'augment_data.py': 'Data augmentation script'
    }
    
    all_exist = True
    
    for file, description in files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {description}")
        else:
            print(f"   ❌ {file} - MISSING ({description})")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some files are missing.")
        print("   Run: python download_model.py for instructions")
        return False
    
    print("✅ All files present!\n")
    return True

def check_webcam():
    """Test if webcam is accessible"""
    print("📷 Testing webcam access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ Webcam not accessible")
            print("   • Check if webcam is connected")
            print("   • Close other applications using webcam")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("   ❌ Could not read frame from webcam")
            return False
        
        print(f"   ✅ Webcam working (Frame shape: {frame.shape})")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version_str}\n")
        return True
    else:
        print(f"   ❌ Python {version_str} (Requires 3.8+)\n")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("🔥 Checking CUDA support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Available")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
            print(f"   ✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"   ⚠️  CUDA not available (will use CPU)")
            print(f"   💡 For GPU support, install: pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
            return True  # Not a critical error
        
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False

def test_model_load():
    """Try loading the model"""
    print("🧠 Testing model load...")
    
    try:
        import os
        if not os.path.exists('sign_language_model_v2.pth'):
            print("   ⚠️  Model file not found (skipping test)")
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_imports),
        ("CUDA Support", check_cuda),
        ("Project Files", check_files),
        ("Model Load", test_model_load),
        ("Webcam", check_webcam),
    ]   # Define model architecture
        class SignLanguageLSTM(nn.Module):
            def __init__(self, input_size=1662, num_classes=6):
                super(SignLanguageLSTM, self).__init__()
                self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
                self.dropout1 = nn.Dropout(0.2)
                self.lstm2 = nn.LSTM(128, 128, batch_first=True)
                self.dropout2 = nn.Dropout(0.2)
                self.lstm3 = nn.LSTM(128, 64, batch_first=True)
                self.dropout3 = nn.Dropout(0.2)
                self.fc1 = nn.Linear(64, 64)
                self.relu = nn.ReLU()
                self.dropout4 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x, _ = self.lstm1(x)
                x = self.dropout1(x)
                x, _ = self.lstm2(x)
                x = self.dropout2(x)
                x, _ = self.lstm3(x)
                x = self.dropout3(x)
                x = x[:, -1, :]
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout4(x)
                x = self.fc2(x)
                return x
        
        actions = np.load('label_map.npy')
        model = SignLanguageLSTM(input_size=1662, num_classes=len(actions))
        
        checkpoint = torch.load('sign_language_model_v2.pth', map_location='cpu')
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print(f"   ✅ Model loaded successfully (PyTorch)")
        print(f"   ✅ Classes: {len(actions)} - {list(actions)[:3]}...")
        print(f"   ✅ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def main():
    print("="*60)
    print("🔍 SignSync AI - System Check")
    print("="*60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_imports),
        ("Project Files", check_files),
        ("Model Load", test_model_load),
        ("Webcam", check_webcam),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Unexpected error in {name}: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("✅ All checks passed! You're ready to run:")
        print("   python real_time_inference.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("   See QUICKSTART.md for troubleshooting help")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
