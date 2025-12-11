import numpy as np
import os
import shutil

# --- CONFIGURATION ---
DATA_PATH = 'MP_Data'
TARGET_SEQUENCES = 30  # We want 30 examples for every word
SEQUENCE_LENGTH = 30   # Each example has 30 frames

def augment_frame(frame_data):
    """
    Takes one frame of landmarks (1662 numbers) and applies random variations.
    """
    frame_data = frame_data.copy()
    
    # 1. Random Noise (Simulates camera noise / slight hand shake)
    # Adds random value between -0.02 and 0.02 to every coordinate
    noise = np.random.normal(0, 0.02, frame_data.shape)
    
    # 2. Random Scaling (Simulates user being closer/further from camera)
    # Scales coordinates by 0.95x (smaller) to 1.05x (bigger)
    scale = np.random.uniform(0.95, 1.05)
    
    # Apply logic
    new_data = (frame_data * scale) + noise
    return new_data

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' not found. Run process_data.py first.")
        return

    # Get all action folders (e.g. "HELLO", "THANKS")
    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    
    print(f"Found {len(actions)} classes. Checking for augmentation needs...")

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        # List existing sequences (0, 1, 2...)
        existing_seqs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        existing_count = len(existing_seqs)
        
        # If we already have enough, skip
        if existing_count >= TARGET_SEQUENCES:
            print(f"Skipping '{action}': Already has {existing_count} sequences.")
            continue
        
        needed = TARGET_SEQUENCES - existing_count
        print(f"Augmenting '{action}': Has {existing_count}, generating {needed} fake sequences...")
        
        # We will clone the FIRST sequence (folder '0') to make new ones
        # (If you have multiple real videos, you could randomly pick one to clone, 
        # but defaulting to '0' is safest if you only have 1 video).
        source_seq_id = existing_seqs[0]
        source_path = os.path.join(action_path, source_seq_id)
        
        # Load the source sequence into memory (list of 30 numpy arrays)
        source_frames = []
        try:
            for frame_num in range(SEQUENCE_LENGTH):
                res = np.load(os.path.join(source_path, f"{frame_num}.npy"))
                source_frames.append(res)
        except Exception as e:
            print(f"  Error loading source sequence {source_seq_id}: {e}")
            continue

        # Generate new sequences
        for i in range(needed):
            # Determine new folder name (e.g., if we have 0, start at 1)
            new_id = existing_count + i
            new_path = os.path.join(action_path, str(new_id))
            os.makedirs(new_path, exist_ok=True)
            
            # Generate 30 augmented frames for this new sequence
            for frame_num, frame_data in enumerate(source_frames):
                # Augment this specific frame
                aug_data = augment_frame(frame_data)
                
                # Save
                np.save(os.path.join(new_path, f"{frame_num}.npy"), aug_data)
                
    print("\nAugmentation Complete! All classes now have 30 sequences.")

if __name__ == "__main__":
    main()