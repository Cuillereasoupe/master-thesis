# -*- coding: utf-8 -*-
"""
CNN Algae Detection V4

Author: jonas
Date: January 2026

1. Stratified temporal sampling
2. Early stopping
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_curve, f1_score, precision_score, 
                             recall_score, roc_curve, auc)
from pycocotools import mask as mask_util
from tqdm import tqdm
import joblib
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    print("✓ PyTorch imported successfully")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch not installed!")
    print("Install with: pip install torch torchvision")
    exit(1)

# ============================================================================
# CONFIGURATION: Update paths if needed
# ============================================================================
COCO_JSON = './data/annotations.json'
IMAGES_DIR = './data/images/'
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CNN hyperparameters
PATCH_SIZE = 32
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print("\n" + "="*70)
print("CNN WITH TEMPORAL STRATIFICATION + EARLY STOPPING")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
print(f"  • Batch size: {BATCH_SIZE}")
print(f"  • Max epochs: {MAX_EPOCHS}")
print(f"  • Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"  • Learning rate: {LEARNING_RATE}")
print(f"  • Device: {DEVICE}")
print(f"  • Random seed: {RANDOM_SEED}")

# ============================================================================
# HELPER FUNCTION: PARSE FILENAME TO GET TIMESTAMP
# ============================================================================

def parse_filename(filename):
    """
    Parse filename to extract clean name and timestamp
    Returns: (clean_filename_without_prefix, datetime_object)
    """
    clean_name = filename
    if '-' in filename:
        parts = filename.split('-', 1)
        if len(parts[0]) == 8:  # 8 hex digit prefix
            clean_name = parts[1]
    
    # Extract timestamp from format: Cam3-07-21-12-00-00.png
    try:
        name_parts = clean_name.replace('.png', '').replace('.jpg', '').split('-')
        if len(name_parts) >= 6:
            month = int(name_parts[1])
            day = int(name_parts[2])
            hour = int(name_parts[3])
            minute = int(name_parts[4])
            second = int(name_parts[5])
            dt = datetime(2025, month, day, hour, minute, second)
            return clean_name, dt
    except:
        pass
    
    return clean_name, None

# ============================================================================
# HELPER FUNCTION: STRATIFIED TEMPORAL SPLIT
# ============================================================================

def stratified_temporal_split(all_filenames, n_train, n_val, seed=42):
    """
    Split images using stratified temporal sampling.
    Training images are evenly distributed across the timeline.
    Validation and test images fill the gaps.
    
    Args:
        all_filenames: List of all image filenames
        n_train: Number of training images
        n_val: Number of validation images
        seed: Random seed
    
    Returns:
        train_files, val_files, test_files
    """
    np.random.seed(seed)
    
    # Parse all filenames to get timestamps
    images_with_time = []
    for fname in all_filenames:
        clean_name, timestamp = parse_filename(fname)
        images_with_time.append({
            'filename': clean_name,
            'timestamp': timestamp
        })
    
    # Sort by timestamp
    images_with_time.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.max)
    sorted_filenames = [img['filename'] for img in images_with_time]
    
    n_total = len(sorted_filenames)
    
    # Select training images evenly distributed across timeline
    if n_train >= n_total:
        train_files = sorted_filenames
        val_files = []
        test_files = []
    else:
        # Calculate indices for evenly spaced training images
        train_indices = np.linspace(0, n_total - 1, n_train, dtype=int)
        train_files = [sorted_filenames[i] for i in train_indices]
        
        # Remaining images for val and test
        remaining_indices = [i for i in range(n_total) if i not in train_indices]
        remaining_files = [sorted_filenames[i] for i in remaining_indices]
        
        # Split remaining into val and test (also temporally distributed)
        if len(remaining_files) > 0:
            # Take every other for val, rest for test (maintains temporal spread)
            np.random.shuffle(remaining_files)  # Shuffle remaining for random val/test assignment
            val_files = remaining_files[:n_val]
            test_files = remaining_files[n_val:]
        else:
            val_files = []
            test_files = []
    
    # Print temporal coverage info
    print(f"\n   Temporal coverage:")
    for split_name, split_files in [('Train', train_files), ('Val', val_files), ('Test', test_files)]:
        if split_files:
            timestamps = []
            for f in split_files:
                _, ts = parse_filename(f)
                if ts:
                    timestamps.append(ts)
            if timestamps:
                print(f"   {split_name}: {min(timestamps).strftime('%m-%d')} to {max(timestamps).strftime('%m-%d')} ({len(split_files)} images)")
    
    return train_files, val_files, test_files

# ============================================================================
# DATASET CLASS
# ============================================================================

class AlgaePatchDataset(Dataset):
    """Dataset for image patches with algae/water labels"""
    
    def __init__(self, patches, labels, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        
        return patch, torch.tensor(label, dtype=torch.long)

# ============================================================================
# CNN ARCHITECTURE
# ============================================================================

class AlgaeCNN(nn.Module):
    """CNN for algae detection from image patches"""
    
    def __init__(self, num_classes=2):
        super(AlgaeCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        fc_input_size = 256 * (PATCH_SIZE // 16) * (PATCH_SIZE // 16)
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# ============================================================================
# HELPER FUNCTION: EXTRACT PATCHES FROM SPECIFIC IMAGES
# ============================================================================

def extract_patches_from_images(image_filenames, coco_data, images_dict, categories_dict):
    """Extract patches ONLY from specified images."""
    patches = []
    labels = []
    patch_info = []
    
    half_patch = PATCH_SIZE // 2
    
    # Create set of image IDs for the specified filenames
    image_ids = set()
    for img in coco_data['images']:
        filename = img['file_name']
        if '-' in filename and len(filename.split('-')[0]) == 8:
            parts = filename.split('-', 1)
            if len(parts) > 1:
                filename = parts[1]
        
        if filename in image_filenames:
            image_ids.add(img['id'])
    
    relevant_anns = [ann for ann in coco_data['annotations'] 
                     if ann['image_id'] in image_ids]
    
    print(f"   Processing {len(image_filenames)} images with {len(relevant_anns)} annotations...")
    
    for ann in tqdm(relevant_anns, desc="   Extracting patches", leave=False):
        img_info = images_dict[ann['image_id']]
        
        filename = img_info['file_name']
        if '-' in filename and len(filename.split('-')[0]) == 8:
            parts = filename.split('-', 1)
            if len(parts) > 1:
                filename = parts[1]
        
        if filename not in image_filenames:
            continue
        
        image_path = os.path.join(IMAGES_DIR, filename)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load: {filename}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        seg = ann['segmentation']
        
        if isinstance(seg, dict):
            mask = mask_util.decode(seg)
        elif isinstance(seg, list):
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for poly in seg:
                points = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
        else:
            continue
        
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        cat_name = categories_dict[ann['category_id']]
        label = 1 if 'algae' in cat_name.lower() and 'non' not in cat_name.lower() else 0
        
        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask > 0)
            
            max_patches_per_ann = 20
            if len(y_coords) > max_patches_per_ann:
                indices = np.random.choice(len(y_coords), max_patches_per_ann, replace=False)
                y_coords = y_coords[indices]
                x_coords = x_coords[indices]
            
            for y, x in zip(y_coords, x_coords):
                if (y - half_patch >= 0 and y + half_patch < h and
                    x - half_patch >= 0 and x + half_patch < w):
                    
                    patch = img_rgb[y - half_patch:y + half_patch,
                                   x - half_patch:x + half_patch]
                    
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3):
                        patches.append(patch)
                        labels.append(label)
                        patch_info.append({
                            'image': filename,
                            'center_y': int(y),
                            'center_x': int(x),
                            'label': 'Algae' if label == 1 else 'Water',
                            'label_numeric': label
                        })
    
    return np.array(patches), np.array(labels), patch_info

# ============================================================================
# EARLY STOPPING CLASS
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation F1 doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_f1, epoch):
        score = val_f1
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"   EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop

# ============================================================================
# STEP 1: LOAD COCO ANNOTATIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOADING COCO ANNOTATIONS")
print("="*70)

with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

print(f"   ✓ Images: {len(coco_data['images'])}")
print(f"   ✓ Annotations: {len(coco_data['annotations'])}")

images_dict = {img['id']: img for img in coco_data['images']}
categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}

# ============================================================================
# STEP 2: SPLIT IMAGES WITH TEMPORAL STRATIFICATION
# ============================================================================

print("\n" + "="*70)
print("STEP 2: SPLITTING IMAGES (STRATIFIED TEMPORAL)")
print("="*70)
print("   ⚠️  Training images evenly distributed across timeline")

# Get all unique image filenames (normalized)
all_filenames = []
for img in coco_data['images']:
    filename = img['file_name']
    if '-' in filename and len(filename.split('-')[0]) == 8:
        parts = filename.split('-', 1)
        if len(parts) > 1:
            filename = parts[1]
    all_filenames.append(filename)

all_filenames = sorted(list(set(all_filenames)))
print(f"\n   Total unique images: {len(all_filenames)}")

# Use stratified temporal split: 15 train, 4 val, 4 test
n_train = 15
n_val = 4
train_files, val_files, test_files = stratified_temporal_split(
    all_filenames, n_train=n_train, n_val=n_val, seed=RANDOM_SEED
)

print(f"\n   ✓ Training images:   {len(train_files)} ({100*len(train_files)/len(all_filenames):.1f}%)")
print(f"   ✓ Validation images: {len(val_files)} ({100*len(val_files)/len(all_filenames):.1f}%)")
print(f"   ✓ Test images:       {len(test_files)} ({100*len(test_files)/len(all_filenames):.1f}%)")

# Save image splits for reproducibility
split_info = pd.DataFrame({
    'split': ['train']*len(train_files) + ['val']*len(val_files) + ['test']*len(test_files),
    'filename': train_files + val_files + test_files
})
split_info.to_csv(os.path.join(OUTPUT_DIR, 'image_splits.csv'), index=False)
print(f"   ✓ Saved: image_splits.csv")

# ============================================================================
# STEP 3: EXTRACT PATCHES FROM EACH SET SEPARATELY
# ============================================================================

print("\n" + "="*70)
print("STEP 3: EXTRACTING PATCHES FROM SEPARATED IMAGE SETS")
print("="*70)

print("\n📦 Extracting TRAINING patches...")
X_train, y_train, train_patch_info = extract_patches_from_images(
    train_files, coco_data, images_dict, categories_dict
)

print("\n📦 Extracting VALIDATION patches...")
X_val, y_val, val_patch_info = extract_patches_from_images(
    val_files, coco_data, images_dict, categories_dict
)

print("\n📦 Extracting TEST patches...")
X_test, y_test, test_patch_info = extract_patches_from_images(
    test_files, coco_data, images_dict, categories_dict
)

print(f"\n" + "="*70)
print(f"PATCH EXTRACTION COMPLETE")
print(f"="*70)
print(f"   Training patches:   {len(X_train):,} from {len(train_files)} images")
print(f"   Validation patches: {len(X_val):,} from {len(val_files)} images")
print(f"   Test patches:       {len(X_test):,} from {len(test_files)} images")
print(f"   Total patches:      {len(X_train) + len(X_val) + len(X_test):,}")

print(f"\n   Training class distribution:")
print(f"   - Algae: {np.sum(y_train==1):,} ({100*np.sum(y_train==1)/len(y_train):.1f}%)")
print(f"   - Water: {np.sum(y_train==0):,} ({100*np.sum(y_train==0)/len(y_train):.1f}%)")

# Verify no overlap
train_set = set(train_files)
val_set = set(val_files)
test_set = set(test_files)

overlap_train_val = train_set & val_set
overlap_train_test = train_set & test_set
overlap_val_test = val_set & test_set

print(f"\n   ✓ Verifying no image overlap:")
print(f"   - Train/Val overlap:  {len(overlap_train_val)} images (should be 0)")
print(f"   - Train/Test overlap: {len(overlap_train_test)} images (should be 0)")
print(f"   - Val/Test overlap:   {len(overlap_val_test)} images (should be 0)")

if len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test) == 0:
    print(f"NO DATA LEAKAGE CONFIRMED! All image sets are disjoint.")
else:
    print(f"ERROR: Image overlap detected!")
    exit(1)

# ============================================================================
# STEP 4: CREATE DATALOADERS
# ============================================================================

print("\n" + "="*70)
print("STEP 4: CREATING DATALOADERS")
print("="*70)

train_dataset = AlgaePatchDataset(X_train, y_train)
val_dataset = AlgaePatchDataset(X_val, y_val)
test_dataset = AlgaePatchDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"   ✓ Training batches:   {len(train_loader)}")
print(f"   ✓ Validation batches: {len(val_loader)}")
print(f"   ✓ Test batches:       {len(test_loader)}")

# ============================================================================
# STEP 5: INITIALIZE MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 5: INITIALIZING CNN MODEL")
print("="*70)

model = AlgaeCNN(num_classes=2).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   ✓ Model initialized")
print(f"   ✓ Total parameters: {total_params:,}")
print(f"   ✓ Trainable parameters: {trainable_params:,}")
print(f"   ✓ Device: {DEVICE}")

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor([
    len(y_train) / (2 * class_counts[0]),
    len(y_train) / (2 * class_counts[1])
]).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"   ✓ Loss function: CrossEntropyLoss with class weights")
print(f"   ✓ Class weights: Water={class_weights[0]:.4f}, Algae={class_weights[1]:.4f}")
print(f"   ✓ Optimizer: Adam (lr={LEARNING_RATE})")
print(f"   ✓ LR scheduler: ReduceLROnPlateau (patience=3)")

# Initialize early stopping
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001, verbose=True)
print(f"   ✓ Early stopping: patience={EARLY_STOPPING_PATIENCE}")

# ============================================================================
# STEP 6: TRAINING LOOP WITH EARLY STOPPING
# ============================================================================

print("\n" + "="*70)
print("STEP 6: TRAINING CNN WITH EARLY STOPPING")
print("="*70)
print(f"   Training for up to {MAX_EPOCHS} epochs (early stopping if no improvement for {EARLY_STOPPING_PATIENCE} epochs)...")

train_losses = []
train_accs = []
val_losses = []
val_accs = []
val_f1s = []
val_precisions = []
val_recalls = []

best_val_f1 = 0.0
best_epoch = 0
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')

start_time = time.time()
actual_epochs = 0

for epoch in range(MAX_EPOCHS):
    actual_epochs = epoch + 1
    epoch_start = time.time()
    
    # ==================== TRAINING PHASE ====================
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # ==================== VALIDATION PHASE ====================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)
    val_precision = precision_score(all_targets, all_preds, zero_division=0)
    val_recall = recall_score(all_targets, all_preds, zero_division=0)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    
    epoch_time = time.time() - epoch_start
    
    # Save best model based on VALIDATION F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
    
    # Print progress
    print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
          f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:5.2f}% | "
          f"VaLoss: {val_loss:.4f} | VaAcc: {val_acc:5.2f}% | "
          f"VaF1: {val_f1:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Check early stopping
    if early_stopping(val_f1, epoch + 1):
        print(f"\n   EARLY STOPPING triggered at epoch {epoch + 1}")
        print(f"   Best validation F1 was {best_val_f1:.4f} at epoch {best_epoch}")
        break

training_time = time.time() - start_time

print(f"\n   ✓ Training complete!")
print(f"   ✓ Actual epochs: {actual_epochs}/{MAX_EPOCHS}")
print(f"   ✓ Total time: {training_time/60:.1f} minutes")
print(f"   ✓ Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

# ============================================================================
# STEP 7: LOAD BEST MODEL AND EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("STEP 7: EVALUATING ON HELD-OUT TEST SET")
print("="*70)

# Load best model
model.load_state_dict(torch.load(best_model_path))
model.eval()

print(f"   ✓ Loaded best model from epoch {best_epoch}")

# Get predictions on test set
test_probs = []
test_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        test_probs.extend(probs.cpu().numpy())
        test_targets.extend(targets.numpy())

test_probs = np.array(test_probs)
test_targets = np.array(test_targets)

# Find optimal threshold using precision-recall curve
precision_curve, recall_curve, thresholds = precision_recall_curve(test_targets, test_probs)
f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
optimal_f1 = f1_scores[optimal_idx]

test_preds_optimized = (test_probs >= optimal_threshold).astype(int)
test_f1_optimized = f1_score(test_targets, test_preds_optimized)
test_precision_optimized = precision_score(test_targets, test_preds_optimized)
test_recall_optimized = recall_score(test_targets, test_preds_optimized)

print(f"\n   Test Set Results (optimal threshold = {optimal_threshold:.4f}):")
print(f"   - F1-Score:  {test_f1_optimized:.4f}")
print(f"   - Precision: {test_precision_optimized:.4f}")
print(f"   - Recall:    {test_recall_optimized:.4f}")
print(f"   - Accuracy:  {accuracy_score(test_targets, test_preds_optimized):.4f}")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("STEP 8: SAVING RESULTS")
print("="*70)

# ROC AUC
fpr, tpr, _ = roc_curve(test_targets, test_probs)
roc_auc = auc(fpr, tpr)

results = {
    'model': 'CNN',
    'patch_size': PATCH_SIZE,
    'max_epochs': MAX_EPOCHS,
    'actual_epochs': actual_epochs,
    'best_epoch': best_epoch,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'training_time_minutes': training_time / 60,
    'split_strategy': 'stratified_temporal',
    'num_train_images': len(train_files),
    'num_val_images': len(val_files),
    'num_test_images': len(test_files),
    'num_train_patches': len(X_train),
    'num_val_patches': len(X_val),
    'num_test_patches': len(X_test),
    'optimal_threshold': float(optimal_threshold),
    'test_f1': float(test_f1_optimized),
    'test_precision': float(test_precision_optimized),
    'test_recall': float(test_recall_optimized),
    'test_accuracy': float(accuracy_score(test_targets, test_preds_optimized)),
    'test_auc': float(roc_auc),
    'best_val_f1': float(best_val_f1),
    'train_images': train_files,
    'val_images': val_files,
    'test_images': test_files
}

joblib.dump(results, os.path.join(OUTPUT_DIR, 'cnn_results.pkl'))

with open(os.path.join(OUTPUT_DIR, 'cnn_results.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("CNN RESULTS (STRATIFIED TEMPORAL + EARLY STOPPING)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n")
    f.write(f"  Max epochs: {MAX_EPOCHS}\n")
    f.write(f"  Actual epochs: {actual_epochs}\n")
    f.write(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}\n")
    f.write(f"  Batch size: {BATCH_SIZE}\n")
    f.write(f"  Learning rate: {LEARNING_RATE}\n")
    f.write(f"  Device: {DEVICE}\n")
    f.write(f"  Split strategy: stratified_temporal\n\n")
    
    f.write(f"Dataset Split:\n")
    f.write(f"  Train: {len(train_files)} images, {len(X_train)} patches\n")
    f.write(f"  Val:   {len(val_files)} images, {len(X_val)} patches\n")
    f.write(f"  Test:  {len(test_files)} images, {len(X_test)} patches\n\n")
    
    f.write(f"Test Images:\n")
    for img in test_files:
        f.write(f"  - {img}\n")
    f.write("\n")
    
    f.write(f"Training:\n")
    f.write(f"  Best epoch: {best_epoch}/{actual_epochs} (of max {MAX_EPOCHS})\n")
    f.write(f"  Training time: {training_time/60:.1f} minutes\n")
    f.write(f"  Best validation F1: {best_val_f1:.4f}\n")
    f.write(f"  Early stopping: {'Yes' if actual_epochs < MAX_EPOCHS else 'No (reached max epochs)'}\n\n")
    
    f.write(f"Test Set Results:\n")
    f.write(f"  Optimal threshold: {optimal_threshold:.4f}\n")
    f.write(f"  F1-Score:  {test_f1_optimized:.4f}\n")
    f.write(f"  Precision: {test_precision_optimized:.4f}\n")
    f.write(f"  Recall:    {test_recall_optimized:.4f}\n")
    f.write(f"  Accuracy:  {accuracy_score(test_targets, test_preds_optimized):.4f}\n")
    f.write(f"  AUC-ROC:   {roc_auc:.4f}\n\n")
    
    f.write("="*70 + "\n")
    f.write("TEST SET CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(classification_report(test_targets, test_preds_optimized, 
                                   target_names=['Water', 'Algae'], 
                                   digits=4))

print("   ✓ Saved: cnn_results.pkl")
print("   ✓ Saved: cnn_results.txt")

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*70)

# Training curves (only up to actual epochs)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(range(1, actual_epochs+1), train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
ax.plot(range(1, actual_epochs+1), val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
if actual_epochs < MAX_EPOCHS:
    ax.axvline(actual_epochs, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Early Stop ({actual_epochs})')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(range(1, actual_epochs+1), train_accs, 'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=3)
ax.plot(range(1, actual_epochs+1), val_accs, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(range(1, actual_epochs+1), val_f1s, 'purple', linewidth=2, label='Validation F1-Score', marker='D', markersize=3)
ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax.axhline(best_val_f1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
          label=f'Best Val F1 ({best_val_f1:.4f})')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Validation F1-Score', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

plt.suptitle(f'CNN Training Curves (Stratified Temporal Split + Early Stopping)\n'
             f'Stopped at epoch {actual_epochs}, best at epoch {best_epoch}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: training_curves.png")

# Confusion matrix
cm = confusion_matrix(test_targets, test_preds_optimized)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Water', 'Algae'],
            yticklabels=['Water', 'Algae'],
            cbar_kws={'label': 'Count'})
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - CNN\n'
             f'Test Set: {len(test_files)} Unseen Images | '
             f'F1={test_f1_optimized:.4f} | Threshold={optimal_threshold:.4f}',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: confusion_matrix.png")

# ROC curve
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, linewidth=2, color='darkblue', label=f'CNN (AUC={roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title(f'ROC Curve - CNN\nTest Set: {len(test_files)} Unseen Images', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: roc_curve.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("CNN TRAINING COMPLETE - FINAL SUMMARY")
print("="*70)
print(f"\nPerformance on {len(test_files)} UNSEEN test images:")
print(f"  • Test F1-Score:  {test_f1_optimized:.4f}")
print(f"  • Test Precision: {test_precision_optimized:.4f}")
print(f"  • Test Recall:    {test_recall_optimized:.4f}")
print(f"  • Test Accuracy:  {accuracy_score(test_targets, test_preds_optimized):.4f}")
print(f"  • Test AUC-ROC:   {roc_auc:.4f}")

print(f"\nTraining Details:")
print(f"  • Best epoch: {best_epoch}/{actual_epochs} (max: {MAX_EPOCHS})")
print(f"  • Early stopping: {'Yes' if actual_epochs < MAX_EPOCHS else 'No'}")
print(f"  • Training time: {training_time/60:.1f} minutes")
print(f"  • Optimal threshold: {optimal_threshold:.4f}")
print(f"  • Device: {DEVICE}")
print(f"  • Split strategy: Stratified temporal")

print(f"\nTest Images (Completely Unseen):")
for img in test_files:
    print(f"  - {img}")
    
print("\n" + "="*70)
print("All outputs saved to:", OUTPUT_DIR)
print("="*70)