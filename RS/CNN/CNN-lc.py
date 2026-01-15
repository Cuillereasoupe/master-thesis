# -*- coding: utf-8 -*-
"""
CNN Learning Curve Analysis - Optimal Dataset Size

Author: jonas
Date: November 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pycocotools import mask as mask_util
from tqdm import tqdm
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    print("✓ PyTorch imported successfully")
except ImportError:
    print("❌ PyTorch not installed!")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
COCO_JSON = './data/result_coco.json'
IMAGES_DIR = './data/images/'
OUTPUT_DIR = './output/learning_curve/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CNN hyperparameters
PATCH_SIZE = 32
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# Learning curve: test with different numbers of training images
N_IMAGES_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
N_REPEATS = 3  # Repeat each for stability

# TEMPORAL STRATEGY: Choose one
# 'chronological': Train on earliest images, test on latest (realistic for future prediction)
# 'stratified_temporal': Train on X images evenly distributed in time, test on the rest (RECOMMENDED)
# 'stratified': Ensure train/test have similar temporal distribution (complex binning)
# 'random': Random split (original behavior, not recommended)
TEMPORAL_STRATEGY = 'stratified_temporal'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print("\n" + "="*70)
print("CNN LEARNING CURVE ANALYSIS (FIXED)")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Training set sizes: {N_IMAGES_LIST}")
print(f"  • Repeats per size: {N_REPEATS}")
print(f"  • Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
print(f"  • Max epochs: {MAX_EPOCHS}")
print(f"  • Device: {DEVICE}")
print(f"  • Temporal strategy: {TEMPORAL_STRATEGY}")
print(f"\nFIXES APPLIED:")
print(f"  ✓ Temporal distribution of images")
print(f"  ✓ Consistent filename parsing")
print(f"  ✓ Clear train/test separation")
print(f"  ✓ Temporal coverage reporting")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """
    Parse filename to extract clean name and timestamp
    JSON has prefix: "e0908920-Cam3-07-21-12-00-00.png"
    Actual files don't: "Cam3-07-21-12-00-00.png"
    Returns: (clean_filename_without_prefix, datetime_object)
    """
    # Remove prefix if present (8 hex digits followed by hyphen)
    clean_name = filename
    if '-' in filename:
        parts = filename.split('-', 1)
        # Check if first part is 8 characters (hex prefix like e9cc6a5d)
        if len(parts[0]) == 8:
            clean_name = parts[1]
    
    # Extract timestamp from format: Cam3-07-21-12-00-00.png
    # Format: CamX-MM-DD-HH-MM-SS.png
    try:
        name_parts = clean_name.replace('.png', '').replace('.jpg', '').split('-')
        if len(name_parts) >= 6:
            # Assuming 2025 and format: MM-DD-HH-MM-SS
            month = int(name_parts[1])
            day = int(name_parts[2])
            hour = int(name_parts[3])
            minute = int(name_parts[4])
            second = int(name_parts[5])
            
            # Create datetime (assuming 2025)
            dt = datetime(2025, month, day, hour, minute, second)
            return clean_name, dt
    except:
        pass
    
    return clean_name, None

def get_image_info_dict(coco_data):
    """
    Create mapping from clean filename (without prefix) to image info
    JSON has: "e9cc6a5d-Cam4-08-25-12-00-00.png"
    Actual files: "Cam4-08-25-12-00-00.png"
    Maps clean filename -> image data
    """
    image_info = {}
    for img in coco_data['images']:
        clean_name, timestamp = parse_filename(img['file_name'])
        # Use clean_name as key (this matches actual file on disk)
        if clean_name not in image_info:
            image_info[clean_name] = {
                'id': img['id'],
                'info': img,
                'timestamp': timestamp,
                'json_name': img['file_name']  # Keep original for reference
            }
    return image_info

def split_images_temporal(all_images_info, n_train, strategy='stratified_temporal', seed=42):
    """
    Split images according to temporal strategy
    
    Args:
        all_images_info: List of dicts with 'filename', 'timestamp'
        n_train: Number of training images
        strategy: 'chronological', 'stratified_temporal', 'stratified', or 'random'
        seed: Random seed for reproducibility
    
    Returns:
        train_filenames, test_filenames, train_imgs, test_imgs
    """
    np.random.seed(seed)
    
    # Sort by timestamp (None timestamps go to end)
    sorted_images = sorted(all_images_info, 
                          key=lambda x: x['timestamp'] if x['timestamp'] else datetime.max)
    
    if strategy == 'chronological':
        # Train on earliest images, test on latest
        train_imgs = sorted_images[:n_train]
        test_imgs = sorted_images[n_train:]
    
    elif strategy == 'stratified_temporal':
        # Select n_train images evenly distributed across the timeline
        # Test on all remaining images
        n_total = len(sorted_images)
        n_test = n_total - n_train
        
        if n_train >= n_total:
            train_imgs = sorted_images
            test_imgs = []
        else:
            # Calculate step size to evenly sample across timeline
            # We want to pick images at indices: 0, step, 2*step, 3*step, ...
            step = (n_total - 1) / (n_train - 1) if n_train > 1 else 0
            
            train_indices = set()
            for i in range(n_train):
                idx = int(round(i * step))
                # Ensure we don't exceed bounds
                idx = min(idx, n_total - 1)
                train_indices.add(idx)
            
            # If rounding caused duplicates, fill remaining slots
            while len(train_indices) < n_train:
                # Find largest gap and add image in the middle
                sorted_indices = sorted(train_indices)
                max_gap_start = 0
                max_gap_size = 0
                for i in range(len(sorted_indices) - 1):
                    gap = sorted_indices[i+1] - sorted_indices[i]
                    if gap > max_gap_size:
                        max_gap_size = gap
                        max_gap_start = sorted_indices[i]
                
                # Add middle of largest gap
                new_idx = max_gap_start + max_gap_size // 2
                if new_idx not in train_indices and new_idx < n_total:
                    train_indices.add(new_idx)
                else:
                    break  # Safety: prevent infinite loop
            
            train_imgs = [sorted_images[i] for i in sorted(train_indices)]
            test_imgs = [img for i, img in enumerate(sorted_images) if i not in train_indices]
        
    elif strategy == 'stratified':
        # Divide timeline into bins, sample proportionally from each
        n_total = len(sorted_images)
        n_test = n_total - n_train
        
        # Create 5 temporal bins
        bin_size = n_total // 5
        train_imgs = []
        test_imgs = []
        
        for i in range(5):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < 4 else n_total
            bin_images = sorted_images[start_idx:end_idx]
            
            n_train_bin = int(n_train * len(bin_images) / n_total)
            np.random.shuffle(bin_images)
            
            train_imgs.extend(bin_images[:n_train_bin])
            test_imgs.extend(bin_images[n_train_bin:])
        
        # Adjust to exact numbers
        all_imgs = train_imgs + test_imgs
        np.random.shuffle(all_imgs)
        train_imgs = all_imgs[:n_train]
        test_imgs = all_imgs[n_train:]
        
    else:  # random
        np.random.shuffle(sorted_images)
        train_imgs = sorted_images[:n_train]
        test_imgs = sorted_images[n_train:]
    
    train_filenames = [img['filename'] for img in train_imgs]
    test_filenames = [img['filename'] for img in test_imgs]
    
    return train_filenames, test_filenames, train_imgs, test_imgs

# ============================================================================
# DATASET & MODEL (unchanged from original)
# ============================================================================

class AlgaePatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        return patch, torch.tensor(label, dtype=torch.long)

class AlgaeCNN(nn.Module):
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

def extract_patches_from_images(image_filenames, image_info_dict, coco_data, categories_dict):
    """
    Extract patches from specified images
    Uses clean filenames and image_info_dict for proper mapping
    """
    patches = []
    labels = []
    half_patch = PATCH_SIZE // 2
    
    # Get image IDs for selected filenames
    image_ids = set()
    for filename in image_filenames:
        if filename in image_info_dict:
            image_ids.add(image_info_dict[filename]['id'])
    
    relevant_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    for ann in relevant_anns:
        if ann['image_id'] not in [image_info_dict[f]['id'] for f in image_filenames]:
            continue
        
        # Find the clean filename
        filename = None
        for fname in image_filenames:
            if image_info_dict[fname]['id'] == ann['image_id']:
                filename = fname
                break
        
        if filename is None:
            continue
        
        image_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        img_info = image_info_dict[filename]['info']
        
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
    
    return np.array(patches), np.array(labels)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
image_info_dict = get_image_info_dict(coco_data)

# Debug: Show filename mapping
print(f"\n   Example filename mappings:")
for i, (clean_name, info) in enumerate(list(image_info_dict.items())[:3]):
    print(f"      JSON: {info['json_name']}")
    print(f"      File: {clean_name}")
    if i < 2:
        print()

# Verify files exist on disk
missing_files = []
existing_files = []
for filename in list(image_info_dict.keys())[:5]:
    filepath = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(filepath):
        existing_files.append(filename)
    else:
        missing_files.append(filename)

if existing_files:
    print(f"\n   ✓ Sample files found on disk: {len(existing_files)}")
if missing_files:
    print(f"   ⚠️  Sample files NOT found: {missing_files}")

# Create list of all images with timestamps
all_images_info = []
for filename, info in image_info_dict.items():
    # Double-check file exists before adding
    filepath = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(filepath):
        all_images_info.append({
            'filename': filename,
            'timestamp': info['timestamp'],
            'id': info['id']
        })
    else:
        print(f"   ⚠️  Skipping {filename} (not found on disk)")

print(f"   ✓ Total images: {len(all_images_info)}")

# Analyze temporal distribution
valid_timestamps = [img['timestamp'] for img in all_images_info if img['timestamp']]
if valid_timestamps:
    print(f"   ✓ Images with valid timestamps: {len(valid_timestamps)}")
    print(f"   ✓ Date range: {min(valid_timestamps).strftime('%Y-%m-%d %H:%M')} to {max(valid_timestamps).strftime('%Y-%m-%d %H:%M')}")
else:
    print(f"   ⚠️  WARNING: No valid timestamps found! Using random split.")
    TEMPORAL_STRATEGY = 'random'

# ============================================================================
# LEARNING CURVE
# ============================================================================

print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)

all_results = []
temporal_stats = []
total_start = time.time()

for n_train in N_IMAGES_LIST:
    print(f"\n{'='*70}")
    print(f"TRAINING WITH {n_train} IMAGES")
    print(f"{'='*70}")
    
    n_test = len(all_images_info) - n_train
    if n_test < 3:
        continue
    
    repeat_results = []
    
    for repeat in range(N_REPEATS):
        print(f"\n   Repeat {repeat+1}/{N_REPEATS}", end=" ")
        
        # Split images with temporal awareness
        train_filenames, test_filenames, train_imgs, test_imgs = split_images_temporal(
            all_images_info, n_train, TEMPORAL_STRATEGY, RANDOM_SEED + repeat + n_train
        )
        
        # Report temporal distribution
        train_dates = [img['timestamp'] for img in train_imgs if img['timestamp']]
        test_dates = [img['timestamp'] for img in test_imgs if img['timestamp']]
        
        if train_dates and test_dates:
            if TEMPORAL_STRATEGY == 'stratified_temporal':
                # Show that train images are spread across the timeline
                print(f"\n      Train coverage: {min(train_dates).strftime('%m-%d')} to {max(train_dates).strftime('%m-%d')} ({len(train_dates)} images evenly distributed)")
                print(f"      Test:  {len(test_dates)} images across full timeline")
            else:
                print(f"\n      Train: {min(train_dates).strftime('%m-%d')} to {max(train_dates).strftime('%m-%d')}")
                print(f"      Test:  {min(test_dates).strftime('%m-%d')} to {max(test_dates).strftime('%m-%d')}")
        
        # Extract patches from train and test images separately
        X_train, y_train = extract_patches_from_images(train_filenames, image_info_dict, coco_data, categories_dict)
        X_test, y_test = extract_patches_from_images(test_filenames, image_info_dict, coco_data, categories_dict)
        
        print(f"      Patches - Train: {len(X_train)}, Test: {len(X_test)}")
        
        if len(X_train) < 50 or len(X_test) < 50:
            print(f"      ⚠️  Skipping: Insufficient patches")
            continue
        
        train_dataset = AlgaePatchDataset(X_train, y_train)
        test_dataset = AlgaePatchDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = AlgaeCNN(num_classes=2).to(DEVICE)
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor([
            len(y_train) / (2 * class_counts[0]),
            len(y_train) / (2 * class_counts[1])
        ]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_f1 = 0.0
        best_preds = None
        best_targets = None
        no_improve = 0
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            model.eval()
            preds = []
            targets_list = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    preds.extend(predicted.cpu().numpy())
                    targets_list.extend(targets.cpu().numpy())
            
            preds = np.array(preds)
            targets_list = np.array(targets_list)
            f1 = f1_score(targets_list, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_preds = preds.copy()
                best_targets = targets_list.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= 5:
                break
        
        precision = precision_score(best_targets, best_preds, zero_division=0)
        recall = recall_score(best_targets, best_preds, zero_division=0)
        accuracy = accuracy_score(best_targets, best_preds)
        
        repeat_results.append({
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        })
        
        # Store temporal info
        if train_dates and test_dates:
            temporal_stats.append({
                'n_train': n_train,
                'repeat': repeat,
                'train_date_min': min(train_dates),
                'train_date_max': max(train_dates),
                'test_date_min': min(test_dates),
                'test_date_max': max(test_dates),
                'f1': best_f1
            })
        
        print(f"      → F1: {best_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    if len(repeat_results) > 0:
        f1s = [r['f1'] for r in repeat_results]
        all_results.append({
            'n_train_images': n_train,
            'n_test_images': n_test,
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'mean_precision': np.mean([r['precision'] for r in repeat_results]),
            'std_precision': np.std([r['precision'] for r in repeat_results]),
            'mean_recall': np.mean([r['recall'] for r in repeat_results]),
            'std_recall': np.std([r['recall'] for r in repeat_results]),
            'mean_accuracy': np.mean([r['accuracy'] for r in repeat_results]),
            'std_accuracy': np.std([r['accuracy'] for r in repeat_results])
        })
        print(f"\n   Summary: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

total_time = time.time() - total_start

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("RESULTS")
print("="*70)

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUTPUT_DIR, 'learning_curve_results_fixed.csv'), index=False)
print(f"\n✓ Saved: learning_curve_results_fixed.csv")
print("\n" + df[['n_train_images', 'mean_f1', 'std_f1', 'mean_precision', 'mean_recall']].to_string(index=False))

# Save temporal stats
if temporal_stats:
    df_temporal = pd.DataFrame(temporal_stats)
    df_temporal.to_csv(os.path.join(OUTPUT_DIR, 'temporal_distribution.csv'), index=False)
    print(f"\n✓ Saved: temporal_distribution.csv")

# Analysis
improvements = []
for i in range(1, len(df)):
    improvements.append(df.iloc[i]['mean_f1'] - df.iloc[i-1]['mean_f1'])

print("\n" + "-"*70)
print("Marginal Improvements:")
for i, imp in enumerate(improvements):
    print(f"   {df.iloc[i]['n_train_images']} → {df.iloc[i+1]['n_train_images']} images: {imp:+.4f}")

last_imp = improvements[-1] if improvements else 0
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if abs(last_imp) < 0.01:
    print(f"\n✅ Performance PLATEAUED (last improvement: {last_imp:+.4f})")
else:
    print(f"\n⚠️  Still improving (last: {last_imp:+.4f})")

print(f"\nFinal Performance ({df.iloc[-1]['n_train_images']} training images):")
print(f"   F1:        {df.iloc[-1]['mean_f1']:.4f} ± {df.iloc[-1]['std_f1']:.4f}")
print(f"   Precision: {df.iloc[-1]['mean_precision']:.4f} ± {df.iloc[-1]['std_precision']:.4f}")
print(f"   Recall:    {df.iloc[-1]['mean_recall']:.4f} ± {df.iloc[-1]['std_recall']:.4f}")
print(f"   Accuracy:  {df.iloc[-1]['mean_accuracy']:.4f} ± {df.iloc[-1]['std_accuracy']:.4f}")

print(f"\nComputation Time: {total_time/60:.1f} minutes")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Main learning curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# F1 Score plot
ax1.plot(df['n_train_images'], df['mean_f1'], 
        marker='o', linewidth=3, markersize=12, color='darkblue', label='CNN F1')
ax1.fill_between(df['n_train_images'], 
                df['mean_f1'] - df['std_f1'],
                df['mean_f1'] + df['std_f1'],
                alpha=0.3, color='blue', label='±1 Std Dev')

for _, row in df.iterrows():
    ax1.annotate(f"{row['mean_f1']:.3f}", 
                xy=(row['n_train_images'], row['mean_f1']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.axhline(0.793, color='red', linestyle='--', linewidth=2, alpha=0.7, label='XGBoost (baseline)')
ax1.axhline(0.806, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='RF (baseline)')

ax1.set_xlabel('Number of Training Images', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax1.set_title(f'Learning Curve: F1-Score vs Dataset Size\n({TEMPORAL_STRATEGY} split)', 
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_ylim([0.7, 1.0])

# Precision-Recall plot
ax2.errorbar(df['n_train_images'], df['mean_precision'], yerr=df['std_precision'],
            marker='s', linewidth=2.5, markersize=10, label='Precision', capsize=5)
ax2.errorbar(df['n_train_images'], df['mean_recall'], yerr=df['std_recall'],
            marker='^', linewidth=2.5, markersize=10, label='Recall', capsize=5)
ax2.errorbar(df['n_train_images'], df['mean_f1'], yerr=df['std_f1'],
            marker='o', linewidth=2.5, markersize=10, label='F1-Score', capsize=5)

ax2.set_xlabel('Number of Training Images', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Precision, Recall, and F1-Score vs Dataset Size', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim([0.7, 1.0])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curve_fixed.png'), dpi=300, bbox_inches='tight')
plt.close()


print("\n" + "="*70)
print(f"All outputs saved to: {OUTPUT_DIR}")
print("="*70)