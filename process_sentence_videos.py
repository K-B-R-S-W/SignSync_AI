import cv2
import numpy as np
import os
import mediapipe as mp
import re

# --- CONFIGURATION ---
# Folder containing your .mp4 files
VIDEO_FOLDER = 'Raw_Videos'
# Folder where training data will be saved
DATA_PATH = 'MP_Data'
# We normalize every video to exactly 30 frames
SEQUENCE_LENGTH = 30

# --- SETUP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic

# --- HELPER FUNCTIONS ---
def clean_filename(filename):
    """
    Converts 'YOU NAME-[what]_ 1080p.mp4' -> 'YOU_NAME_WHAT'
    """
    name = os.path.splitext(filename)[0]  # Remove extension (.mp4)
    
    # 1. Remove specific junk words (Case insensitive)
    junk_words = ['1080p', '720p', '4k', 'hd', 'sd']
    for junk in junk_words:
        name = re.sub(junk, '', name, flags=re.IGNORECASE)

    # 2. Replace all special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    
    # 3. Clean up multiple underscores and strip edges
    name = re.sub(r'_+', '_', name)
    return name.upper().strip('_')

def extract_keypoints(results):
    """
    Extracts all landmarks (Pose + Face + Hands) into a single flat array.
    Total features: 1662
    """
    # 1. Pose (Body): 33 points * 4 dims (x,y,z,vis)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 2. Face: 468 points * 3 dims (x,y,z)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # 3. Left Hand: 21 points * 3 dims
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 4. Right Hand: 21 points * 3 dims
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Combine everything
    return np.concatenate([pose, face, lh, rh])

# --- MAIN PROCESSING LOOP ---
def main():
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Error: Folder '{VIDEO_FOLDER}' not found. Please create it and add videos.")
        return

    # Get list of video files
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    print(f"Found {len(video_files)} videos. Starting processing...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for video_file in video_files:
            # 1. Generate Label from Filename
            action_name = clean_filename(video_file)
            
            # 2. Create directory for this label
            action_folder = os.path.join(DATA_PATH, action_name)
            os.makedirs(action_folder, exist_ok=True)
            
            # Determine the next sequence number (0, 1, 2...)
            existing_seqs = [int(f) for f in os.listdir(action_folder) if f.isdigit()]
            sequence_num = max(existing_seqs) + 1 if existing_seqs else 0
            
            sequence_path = os.path.join(action_folder, str(sequence_num))
            os.makedirs(sequence_path, exist_ok=True)
            
            print(f"Processing: '{video_file}' -> Class: '{action_name}' (Seq {sequence_num})")

            # 3. Open Video
            cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < SEQUENCE_LENGTH:
                print(f"  Warning: Video '{video_file}' is too short ({total_frames} frames). Skipping.")
                continue

            # 4. Smart Frame Skipping (Pick 30 frames evenly distributed)
            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
            
            current_frame_idx = 0
            saved_count = 0
            
            while cap.isOpened() and saved_count < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: break
                
                # If this frame is in our 'selected' list, process it
                if current_frame_idx in frame_indices:
                    # MediaPipe Detection
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    
                    # Extract & Save
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, f"{saved_count}.npy")
                    np.save(npy_path, keypoints)
                    
                    saved_count += 1
                
                current_frame_idx += 1
                
            cap.release()

    print("\nProcessing Complete!")
    print(f"Data saved in '{DATA_PATH}/'. Run augment_data.py next.")

if __name__ == "__main__":
    main()