import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

DATA_PATH = 'MP_Data'
DRIVE_FILE_PATH = '/content/drive/MyDrive/Sign/MP_Data.rar'
MODEL_SAVE_PATH = '/content/drive/MyDrive/sign_language_model_v2.pth'
LABEL_SAVE_PATH = '/content/drive/MyDrive/label_map.npy'

# Training parameters
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 30
INPUT_SIZE = 1662

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")


# Extract data from Drive
if not os.path.exists('MP_Data'):
    if os.path.exists(DRIVE_FILE_PATH):
        print(f"Found file at {DRIVE_FILE_PATH}. Copying...")
        filename = os.path.basename(DRIVE_FILE_PATH)
        !cp "$DRIVE_FILE_PATH" .
        
        if filename.endswith('.rar'):
            print("Detected .rar file. Installing unrar...")
            !apt-get install unrar > /dev/null
            print("Extracting...")
            !unrar x "$filename" > /dev/null
        else:
            print("Extracting...")
            !unzip -q "$filename"
        
        print(" Data ready!")
    else:
        print(f" ERROR: Could not find file at: {DRIVE_FILE_PATH}")
        raise FileNotFoundError("File not found in Drive.")
else:
    print("Data already extracted. Skipping.")


# Detect classes
actions = np.array([folder for folder in os.listdir(DATA_PATH) 
                    if os.path.isdir(os.path.join(DATA_PATH, folder))])
print(f" Found {len(actions)} classes: {actions}")

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
print("Loading data into memory...")

for action in actions:
    for sequence in range(30):
        window = []
        try:
            for frame_num in range(30):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            print(f" Warning: Missing data in {action}/{sequence}")

X = np.array(sequences, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

print(f" Total samples: {len(X)}")
print(f" Data shape: {X.shape}")

class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
)

print(f" Training samples: {len(X_train)}")
print(f" Validation samples: {len(X_val)}")
print(f" Test samples: {len(X_test)}")

# Create datasets and dataloaders
train_dataset = SignLanguageDataset(X_train, y_train)
val_dataset = SignLanguageDataset(X_val, y_val)
test_dataset = SignLanguageDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


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

# Initialize model
model = SignLanguageLSTM(input_size=INPUT_SIZE, num_classes=len(actions))
model = model.to(device)

print(" Model initialized")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Training history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'lr': []
}

print("\n" + "="*60)
print(" Starting Training (PyTorch + CUDA)")
print("="*60)

best_val_loss = float('inf')
patience = 30
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    current_lr = optimizer.param_groups[0]['lr']
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n Best model from epoch {checkpoint['epoch']+1}")

print("\n" + "="*60)
print(" EVALUATING MODEL ON TEST SET")
print("="*60)

model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct / test_total
print(f"\n Test Accuracy: {test_acc*100:.2f}%")
print(f" Test Samples: {test_total}")

# Classification report
print("\n" + "="*60)
print(" DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_labels, all_predictions, target_names=actions, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=actions, yticklabels=actions)
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc*100:.1f}%')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Training History
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
axes[0].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(history['lr'], linewidth=2, color='green')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

if test_acc > 0.60:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'num_classes': len(actions),
        'input_size': INPUT_SIZE,
        'sequence_length': SEQUENCE_LENGTH,
    }, MODEL_SAVE_PATH)
    
    np.save(LABEL_SAVE_PATH, actions)
    
    print(f"\n Model saved to: {MODEL_SAVE_PATH}")
    print(f" Label mapping saved to: {LABEL_SAVE_PATH}")
else:
    print(f"\n Model accuracy ({test_acc*100:.1f}%) too low to save.")
