# 🤟 SignSync AI

An advanced pipeline for creating custom Sign Language datasets using **MediaPipe Holistic** and **Computer Vision**.  
This project recognizes **full sentences (dynamic gestures)** from video input and prepares them for **Deep Learning (LSTM)** training.

---

## 🚀 Features

- **Holistic Tracking**  
  Captures Body Pose, Face, and Hands simultaneously for full context (e.g., distinguishes "Please" vs "Sorry").

- **Smart Video Processing**  
  Converts raw video files into normalized landmark `.npy` data automatically.

- **Auto-Augmentation**  
  Generates **30+ variations** from a single video to overcome limited dataset size (noise, scaling, rotation).

- **PyTorch + CUDA GPU Acceleration**  
  Lightning-fast training (10-15 min) and real-time inference (60+ FPS) with GPU support.

- **Real-Time Inference with Auto-Lock**  
  Uses webcam to detect sign language gestures with confidence scores. Predictions auto-lock and stay visible until manually cleared.

- **Motion Detection & Stability**  
  Prevents false predictions on resting poses. Only triggers when actual signing motion is detected.

- **LLM-Ready Labels**  
  Automatically formats filenames like `YOU NAME?.mp4` → `YOU_NAME` for smoother LLM-based sentence reconstruction.

---

## 📂 Project Structure

```
SignSync_AI/
│
├── Raw_Videos/                  <-- [INPUT] Put your MP4 files here
│   ├── HELLO.mp4
│   ├── THANK YOU.mp4
│   └── YOUR TEACHER what-NAME.mp4
│
├── process_sentence_videos.py  <-- Script 1: Extracts landmarks
├── augment_data.py              <-- Script 2: Multiplies data
├── real_time_inference.py       <-- Script 3: Webcam detection (PyTorch)
├── test_setup.py                <-- Verify PyTorch/CUDA installation
│
├── Data_Train/                  <-- Training scripts
│   ├── train_pytorch.py  
│   └── train_Tensor.py        
│
├── Models/                      <-- [OUTPUT] Trained models
│   ├── sign_language_model_v2.pth  <-- PyTorch model weights
│   └── label_map.npy            <-- Class name mappings
│
└── MP_Data/                     <-- [OUTPUT] Generated automatically
    ├── HELLO/                   <-- Class Folder
    │   ├── 0/                   <-- Sequence 0 (Original)
    │   ├── 1/                   <-- Sequence 1 (Augmented)
    │   └── ...
    └── ...
```

## 🛠️ Installation

**Prerequisites:**
- Python 3.8 or higher
- Webcam (for real-time inference)
- Google Drive account (for cloud training)
- **NVIDIA GPU (optional but recommended)** - 10-20x faster training/inference
- CUDA Toolkit 11.8+ (if using GPU)

**Install PyTorch with CUDA (GPU):**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Or CPU-Only Version:**

```bash
pip3 install torch torchvision torchaudio
```

**Install Other Dependencies:**

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib seaborn
```

**Verify Installation:**

```bash
python test_setup.py
```

This will check:
- ✅ PyTorch installed
- ✅ CUDA available (if GPU present)
- ✅ GPU name and CUDA version
- ✅ All required packages

**📖 Detailed Setup Guide:** See `PYTORCH_CUDA_SETUP.md` for troubleshooting and optimization tips.

---

## 📚 Test Run Images















---

## 📚 Documentation

- **`PYTORCH_CUDA_SETUP.md`** - Complete PyTorch + CUDA installation guide
- **`QUICKSTART.md`** - Fast-track guide for experienced users
- **`INFERENCE_GUIDE.md`** - Technical details on real-time detection

---

⚡ How to Use

### **Phase 1: Data Preparation**

**Step 1: Add Your Videos**

Place your recorded sign language videos into the `Raw_Videos/` folder.

Naming Convention:
- ✔️ **Good:** `I_LOVE_YOU.mp4`, `WHERE_IS_BATHROOM.mp4`
- ❌ **Bad:** `VID_2023.mp4`

**Step 2: Extract Landmarks**

Converts each video into mathematical keypoints — the video itself is discarded, keeping only skeletal data.

```bash
python process_sentence_videos.py
```

**Output:** Creates `MP_Data/` with sequence 0 for each video.

**Step 3: Augment Data**

Generates 29 augmented variations per video (zoomed, shaky, rotated, etc.).

```bash
python augment_data.py
```

**Output:** Each class folder in `MP_Data/` will now contain sequences 0 → 29.

---

### **Phase 2: Model Training**

**Step 4: Upload Data to Google Drive**

1. Compress the `MP_Data/` folder:
   ```bash
   # Windows: Right-click → Send to → Compressed folder
   # Or use 7-Zip/WinRAR to create MP_Data.rar
   ```

2. Upload `MP_Data.rar` to your Google Drive in this path:
   ```
   Google Drive/Sign/MP_Data.rar
   ```

**Step 5: Train with PyTorch on Google Colab**

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/)
2. Upload `Data_Train/train_pytorch.py` or copy code into a new notebook
3. Enable GPU: `Runtime` → `Change runtime type` → `GPU` → `T4/V100`
4. Run all cells

**Training Features:**
- 🚀 **GPU Acceleration**: Tesla T4 or better
- 📊 **Real-time Metrics**: Loss, accuracy, learning rate
- 🎯 **Early Stopping**: Prevents overfitting (patience=30)
- 📉 **LR Scheduling**: Adaptive learning rate reduction
- 📈 **Visualization**: Confusion matrix, training curves
- 💾 **Auto-Save**: Best model saved to Google Drive

**Expected Results:**
- Training time: ~10-15 minutes on GPU (vs 90+ min on CPU)
- Target accuracy: 85-100% (depends on dataset quality)
- Model saves automatically to `Google Drive/sign_language_model_v2.pth`

**Output Files:**
- `sign_language_model_v2.pth` - PyTorch model weights
- `label_map.npy` - Class name mappings

🧠 Roadmap

- ✅ **Dataset Creation Pipeline** - Extract + Augment landmark data
- ✅ **PyTorch LSTM Training** - GPU-accelerated with CUDA support
- ✅ **Real-Time Inference** - Auto-lock predictions, motion detection
- 🔄 **LLM Integration** - Convert glosses to natural language (Coming Soon)
  - Example: `YOU NAME WHAT` → "What is your name?"
- 🔄 **Web UI** - Streamlit/Gradio interface (Coming Soon)
- 🔄 **Mobile App** - Deploy to mobile devices (Future)

---

### **Phase 3: Real-Time Detection**

**Step 6: Download Trained Model**

Download from Google Drive:
1. `sign_language_model_v2.pth` (PyTorch model)
2. `label_map.npy` (class mappings)

Place both files in the `Models/` folder:
```
SignSync_AI/
└── Models/
    ├── sign_language_model_v2.pth
    └── label_map.npy
```

**Step 7: Run Real-Time Inference**

```bash
python real_time_inference.py
```

**Controls:**
- **[Q]** - Quit application
- **[R]** - Toggle recording (start/stop frame collection)
- **[C]** - Clear sequence and unlock prediction
- **[L]** - lock/unlock current prediction

**Features:**
- ✅ **GPU Acceleration**: 60+ FPS on RTX cards (vs 20 FPS CPU)
- ✅ **Auto-Lock System**: Predictions lock after detection and stay visible
- ✅ **Motion Detection**: Ignores static/resting poses
- ✅ **Prediction Stability**: Requires 2/3 consecutive frames for confirmation
- ✅ **Visual Feedback**: 
  - 🟢 Green = Active detection
  - 🔵 Blue = Locked prediction
  - 🔒 Lock icon when prediction held
- ✅ **Live Skeleton Overlay**: MediaPipe landmarks visualization
- ✅ **Confidence Threshold**: 85% minimum (adjustable)
- ✅ **FPS Counter**: Real-time performance monitoring
- ✅ **30-Frame Sequences**: Temporal context for accurate prediction

**Performance:**
| Hardware | Inference Speed | Training Time |
|----------|----------------|---------------|
| CPU (Intel i7) | ~5 FPS | ~90 minutes |
| GPU (RTX 3050) | ~20 FPS | ~10 minutes |
| GPU (RTX 3060+) | ~30 FPS | ~6 minutes |

---

## 🔮 AI Integration (Coming Soon)

### **Text-to-Speech with LLM Enhancement**

Convert sign language predictions into natural spoken language:

**Planned Features:**
- 🗣️ **Text-to-Speech (TTS)**: Real-time voice output of detected signs
- 🤖 **LLM Integration**: Convert sign glosses to grammatically correct sentences
  - Example: `YOU NAME WHAT` → "What is your name?"
  - Example: `I HUNGRY WANT EAT` → "I'm hungry and want to eat"
- 🌐 **Multi-Language Support**: English, Spanish, Tamil etc.
- 🎙️ **Voice Selection**: Male/female voices, accent options
- 📝 **Conversation History**: Track and replay sign language conversations
- 💾 **Export Options**: Save conversations as text/audio files

**Technology Stack:**
- **LLM**: OpenAI GPT-4 / Google Gemini / Local LLaMA
- **TTS Engine**: ElevenLabs / Google TTS / Coqui TTS
- **Pipeline**: Sign Detection → LLM Processing → TTS Output

**Use Cases:**
- 🏫 Educational environments (classroom assistance)
- 🏥 Medical consultations
- 🏢 Professional meetings
- 🛒 Customer service interactions
- 📱 Video call accessibility

Stay tuned for updates! 🚀

