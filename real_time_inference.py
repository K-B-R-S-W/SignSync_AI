import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import os
from collections import deque
import time

# --- CONFIGURATION ---
MODEL_PATH = 'Models/sign_language_model_v2.pth'  # Path to your trained model (PyTorch)
LABEL_PATH = 'Models/label_map.npy'  # Path to your label mapping
SEQUENCE_LENGTH = 30  # Must match training (30 frames)
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence to display prediction (85%)
MOTION_THRESHOLD = 0.01  # Minimum movement required to trigger prediction
STABILITY_FRAMES = 3  # Number of consecutive frames with same prediction needed

# --- CUDA CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# --- SETUP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- HELPER FUNCTIONS ---
def detect_motion(sequence, threshold=MOTION_THRESHOLD):
    """
    Detects if there's significant motion in the sequence.
    Helps prevent false predictions on static/resting poses.
    """
    if len(sequence) < 5:
        return False
    
    # Calculate variance across recent frames (focus on hand keypoints)
    recent_frames = np.array(list(sequence)[-10:])  # Last 10 frames
    
    # Hand keypoints start at index 1662-126 (last 126 values = both hands)
    hand_data = recent_frames[:, -126:]  # Extract hand landmarks
    
    # Calculate standard deviation (motion indicator)
    motion = np.std(hand_data)
    
    return motion > threshold

def extract_keypoints(results):
    """
    Extracts all landmarks (Pose + Face + Hands) into a single flat array.
    Must match the training data format (1662 features)
    """
    # 1. Pose (Body): 33 points * 4 dims (x,y,z,vis)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 2. Face: 468 points * 3 dims (x,y,z)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # 3. Left Hand: 21 points * 3 dims
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 4. Right Hand: 21 points * 3 dims
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    """
    Draws MediaPipe landmarks on the frame with custom styling
    """
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
    )
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
    )
    
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
    )
    

def draw_ui(image, prediction, confidence, fps, recording, is_locked=False):
    """
    Draws UI elements on the frame
    """
    h, w, _ = image.shape
    
    # Semi-transparent overlay for text background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Title
    cv2.putText(image, "SignSync AI - Real-Time Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
    
    # FPS Counter
    cv2.putText(image, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
    
    # Recording status
    if recording:
        status_color = COLOR_GREEN if confidence >= CONFIDENCE_THRESHOLD else COLOR_RED
        cv2.circle(image, (w - 30, 60), 8, status_color, -1)
        cv2.putText(image, "RECORDING", (w - 120, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    # Prediction display
    if prediction and confidence >= CONFIDENCE_THRESHOLD:
        # Clean up prediction text (replace underscores with spaces)
        display_text = prediction.replace('_', ' ')
        
        # Color changes if locked
        bar_color = COLOR_BLUE if is_locked else COLOR_GREEN
        text_color = COLOR_BLUE if is_locked else COLOR_GREEN
        
        # Confidence bar
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(image, (20, 80), (w - 20, 100), (50, 50, 50), -1)
        cv2.rectangle(image, (20, 80), (20 + bar_width, 100), bar_color, -1)
        
        # Prediction text with lock indicator
        lock_icon = "🔒 " if is_locked else ""
        cv2.putText(image, f"{lock_icon}{display_text}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(image, f"{confidence*100:.1f}%", (w - 100, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    else:
        # Waiting message
        cv2.putText(image, "Waiting for gesture...", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    # Instructions at bottom
    cv2.rectangle(image, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(image, "Q: Quit  |  R: Record  |  C: Clear  |  L: Lock/Unlock prediction", 
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    lock_status = "🔒 LOCKED" if is_locked else "Recording..." if recording else "Paused"
    cv2.putText(image, f"Status: {lock_status}", 
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

# --- PYTORCH LSTM MODEL DEFINITION ---
class SignLanguageLSTM(nn.Module):
    """
    PyTorch LSTM model matching the Keras architecture
    """
    def __init__(self, input_size=1662, num_classes=6):
        super(SignLanguageLSTM, self).__init__()
        
        # LSTM Layers
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
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # Take only the last time step
        x = x[:, -1, :]
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

# --- MAIN PROGRAM ---
def main():
    print("="*60)
    print("🎥 SignSync AI - Real-Time Inference (PyTorch + CUDA)")
    print("="*60)
    
    # 1. Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at '{MODEL_PATH}'")
        print("Please ensure the model file is in the same directory.")
        print("Expected: .pth file (PyTorch format)")
        return
    
    print(f"Loading model from '{MODEL_PATH}'...")
    
    # Load label mapping first to get num_classes
    if not os.path.exists(LABEL_PATH):
        print(f"❌ ERROR: Label map not found at '{LABEL_PATH}'")
        return
    
    actions = np.load(LABEL_PATH)
    num_classes = len(actions)
    
    # Initialize model
    model = SignLanguageLSTM(input_size=1662, num_classes=num_classes)
    
    # Load trained weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Note: If you have a Keras .h5 file, you need to convert it to PyTorch .pth")
        return
    
    # Move model to device (GPU/CPU)
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully on {DEVICE}!")
    if DEVICE.type == 'cuda':
        print(f"   🚀 GPU Acceleration: ENABLED")
    
    # 2. Load label mapping
    if not os.path.exists(LABEL_PATH):
        print(f"❌ ERROR: Label map not found at '{LABEL_PATH}'")
        print("Please ensure the label_map.npy file is in the same directory.")
        return
    
    actions = np.load(LABEL_PATH)
    print(f"✅ Loaded {len(actions)} classes: {list(actions)}")
    
    # 3. Initialize webcam
    print("\n📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not access webcam.")
        print("Make sure your webcam is connected and not being used by another application.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Webcam initialized!")
    print("\n" + "="*60)
    print("🚀 Starting real-time detection...")
    print("="*60)
    print("\nControls:")
    print("  [Q] - Quit")
    print("  [R] - Toggle recording (start/stop collecting frames)")
    print("  [C] - Clear sequence and unlock")
    print("  [L] - Manually lock/unlock current prediction")
    print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print("\n💡 TIP: Predictions auto-lock after detection. Press C to clear.\n")
    
    # Initialize variables
    sequence = deque(maxlen=SEQUENCE_LENGTH)  # Rolling buffer of 30 frames
    prediction = ""
    confidence = 0.0
    recording = False
    frame_count = 0
    
    # Prediction stability tracking
    prediction_history = deque(maxlen=STABILITY_FRAMES)
    last_stable_prediction = ""
    last_stable_confidence = 0.0
    
    # Locked prediction (holds result until manually cleared)
    is_locked = False
    locked_prediction = ""
    locked_confidence = 0.0
    
    # FPS calculation
    fps_time = time.time()
    fps = 0
    
    # 4. Start MediaPipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = holistic.process(image)
            
            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Add to sequence if recording
            if recording:
                sequence.append(keypoints)
                frame_count += 1
                
                # Make prediction when we have enough frames
                if len(sequence) == SEQUENCE_LENGTH:
                    # Check if there's actual movement
                    has_motion = detect_motion(sequence)
                    
                    if has_motion:
                        # Prepare input for model (PyTorch)
                        input_data = np.expand_dims(list(sequence), axis=0)
                        input_tensor = torch.FloatTensor(input_data).to(DEVICE)
                        
                        # Predict with GPU acceleration
                        with torch.no_grad():  # Disable gradient calculation for inference
                            output = model(input_tensor)
                            res = output.cpu().numpy()[0]  # Move back to CPU for numpy operations
                        
                        predicted_class = np.argmax(res)
                        temp_confidence = res[predicted_class]
                        temp_prediction = actions[predicted_class]
                        
                        # Add to prediction history
                        prediction_history.append(temp_prediction)
                        
                        # Check if prediction is stable (appears multiple times consecutively)
                        if len(prediction_history) == STABILITY_FRAMES:
                            # Count most common prediction
                            unique, counts = np.unique(list(prediction_history), return_counts=True)
                            most_common_idx = np.argmax(counts)
                            most_common = unique[most_common_idx]
                            
                            # Only update if prediction is stable and confident
                            if counts[most_common_idx] >= 2 and temp_confidence >= CONFIDENCE_THRESHOLD:
                                # If not locked, update normally
                                if not is_locked:
                                    prediction = temp_prediction
                                    confidence = temp_confidence
                                    
                                    # Only print if prediction changed
                                    if prediction != last_stable_prediction:
                                        last_stable_prediction = prediction
                                        last_stable_confidence = confidence
                                        device_label = "GPU" if DEVICE.type == 'cuda' else "CPU"
                                        print(f"✅ [{device_label}] Detected: {prediction.replace('_', ' '):30s} | Confidence: {confidence*100:5.1f}%")
                                        
                                        # Auto-lock after confident detection
                                        is_locked = True
                                        locked_prediction = prediction
                                        locked_confidence = confidence
                    else:
                        # No motion detected
                        if len(prediction_history) > 0:
                            prediction_history.clear()
            
            # Use locked prediction if active
            display_prediction = locked_prediction if is_locked else prediction
            display_confidence = locked_confidence if is_locked else confidence
            # Calculate FPS
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            # Draw UI
            draw_ui(image, display_prediction, display_confidence, fps, recording, is_locked)
            
            # Display
            cv2.imshow('SignSync AI - Real-Time Detection', image)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Exiting...")
                break
            elif key == ord('r') or key == ord('R'):
                recording = not recording
                if recording:
                    print("▶️  Recording started - collecting frames...")
                    sequence.clear()
                    frame_count = 0
                else:
                    print("⏸️  Recording paused")
            elif key == ord('c') or key == ord('C'):
                sequence.clear()
                prediction = ""
                confidence = 0.0
                frame_count = 0
                is_locked = False
                locked_prediction = ""
                locked_confidence = 0.0
                prediction_history.clear()
                print("🗑️  Sequence cleared (unlocked)")
            elif key == ord('l') or key == ord('L'):
                is_locked = not is_locked
                if is_locked:
                    # Lock current prediction
                    locked_prediction = prediction
                    locked_confidence = confidence
                    print(f"🔒 Locked: {prediction.replace('_', ' ')}")
                else:
                    # Unlock
                    locked_prediction = ""
                    locked_confidence = 0.0
                    print("🔓 Unlocked - ready for new detection")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program terminated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
