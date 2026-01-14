# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:20:01 2025

@author: jonas

CNN Full-Image Application for Thesis
Applies trained 32×32 CNN to complete images for algae detection

Key Features:
- Sliding window with overlap for smooth predictions
- Both pixel-level and region-level evaluation
- Timing information for computational cost analysis
- Handles all images (annotated + unannotated)
- Generates thesis-ready visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from glob import glob
from scipy.ndimage import label as connected_components
import time
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
COCO_JSON = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/result_coco.json'
IMAGES_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/transformed/'
OUTPUT_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/CNN/full_image_predictions/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CNN model path (from learning curve training)
CNN_MODEL_PATH = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/CNN/output/best_model.pth'

# CRITICAL: Must match training configuration
PATCH_SIZE = 32  # From learning curve script
STRIDE = 16      # 50% overlap for smoother predictions
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prediction settings
BATCH_SIZE = 128  # Larger batches for faster inference
THRESHOLD = 0.5   # Will be optimized based on validation set

# Processing options
PROCESS_ALL_IMAGES = True  # Set False to only process annotated images
MAX_IMAGES = None  # Set to number to limit processing (None = all)

print("="*70)
print("CNN FULL-IMAGE PREDICTION FOR THESIS")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Patch size: {PATCH_SIZE}×{PATCH_SIZE} (matches training)")
print(f"  • Stride: {STRIDE} pixels (overlap for smooth predictions)")
print(f"  • Batch size: {BATCH_SIZE}")
print(f"  • Device: {DEVICE}")
print(f"  • Threshold: {THRESHOLD} (will optimize if needed)")
print(f"  • Process all images: {PROCESS_ALL_IMAGES}")
print(f"  • Output: {OUTPUT_DIR}")

# ============================================================================
# CNN ARCHITECTURE (MUST MATCH LEARNING CURVE TRAINING)
# ============================================================================

class AlgaeCNN(nn.Module):
    """CNN architecture matching the learning curve training"""
    
    def __init__(self, num_classes=2):
        super(AlgaeCNN, self).__init__()
        
        # 4 convolutional blocks
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
        
        # Calculate FC input size: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        fc_input_size = 256 * (PATCH_SIZE // 16) * (PATCH_SIZE // 16)
        
        # Fully connected layers
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
# LOAD MODEL
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

try:
    model = AlgaeCNN(num_classes=2).to(DEVICE)
    
    # Try to load checkpoint
    if os.path.exists(CNN_MODEL_PATH):
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded model from checkpoint: {CNN_MODEL_PATH}")
            
            # Try to get optimal threshold if saved
            if 'optimal_threshold' in checkpoint:
                THRESHOLD = checkpoint['optimal_threshold']
                print(f"   ✓ Using saved optimal threshold: {THRESHOLD:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded model state dict: {CNN_MODEL_PATH}")
    else:
        print(f"   ⚠️  Model not found at: {CNN_MODEL_PATH}")
        print(f"   Looking for alternative model files...")
        
        # Look for any .pth file in the directory
        model_dir = os.path.dirname(CNN_MODEL_PATH)
        pth_files = glob(os.path.join(model_dir, '*.pth'))
        
        if pth_files:
            alt_model = pth_files[0]
            checkpoint = torch.load(alt_model, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded alternative model: {alt_model}")
        else:
            print(f"   ✗ No model files found in {model_dir}")
            exit(1)
    
    model.eval()
    print(f"   ✓ Model in evaluation mode")
    print(f"   ✓ Device: {DEVICE}")
    
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """Remove prefix from JSON filenames to match disk files"""
    clean_name = filename
    if '-' in filename:
        parts = filename.split('-', 1)
        if len(parts[0]) == 8:  # 8-char hex prefix
            clean_name = parts[1]
    return clean_name

def predict_full_image(image_rgb, model, patch_size=32, stride=16, threshold=0.5, device='cpu'):
    """
    Apply CNN to full image using sliding window with overlap
    
    Args:
        image_rgb: HxWx3 RGB image
        model: Trained CNN model
        patch_size: Size of patches (must match training)
        stride: Step size for sliding window (smaller = more overlap = smoother)
        threshold: Probability threshold for binary classification
        device: 'cuda' or 'cpu'
    
    Returns:
        probability_map: HxW array of algae probabilities (0-1)
        prediction_map: HxW array of binary predictions (0=water, 1=algae)
        inference_time: Time taken for inference (seconds)
    """
    
    start_time = time.time()
    
    h, w, _ = image_rgb.shape
    
    # Initialize probability and count maps for averaging overlaps
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Collect all patches and positions
    patches = []
    positions = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image_rgb[y:y+patch_size, x:x+patch_size]
            
            if patch.shape[:2] == (patch_size, patch_size):
                # Normalize and convert to tensor
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
                patches.append(patch_tensor)
                positions.append((y, x))
    
    # Batch prediction for efficiency
    n_patches = len(patches)
    
    with torch.no_grad():
        for i in range(0, n_patches, BATCH_SIZE):
            batch = torch.stack(patches[i:i+BATCH_SIZE]).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of algae class
            
            # Add to probability map (will be averaged later)
            for j, (y, x) in enumerate(positions[i:i+BATCH_SIZE]):
                prob_map[y:y+patch_size, x:x+patch_size] += probs[j].cpu().numpy()
                count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)
    
    # Apply threshold
    prediction_map = (prob_map >= threshold).astype(np.uint8)
    
    inference_time = time.time() - start_time
    
    return prob_map, prediction_map, inference_time

def compute_pixel_metrics(pred_mask, gt_mask):
    """Compute pixel-level metrics"""
    
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def compute_region_metrics(pred_mask, gt_mask, iou_threshold=0.3):
    """
    Compute region-level metrics (detection of algae blobs)
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
        iou_threshold: Minimum IoU to consider a region as detected
    
    Returns:
        dict with region-level TP, FP, FN, precision, recall, F1
    """
    
    # Label connected components
    gt_labeled, n_gt = connected_components(gt_mask)
    pred_labeled, n_pred = connected_components(pred_mask)
    
    if n_gt == 0 and n_pred == 0:
        return {
            'region_tp': 0, 'region_fp': 0, 'region_fn': 0,
            'region_precision': 1.0, 'region_recall': 1.0, 'region_f1': 1.0,
            'total_gt_regions': 0, 'total_pred_regions': 0
        }
    
    if n_gt == 0:
        return {
            'region_tp': 0, 'region_fp': n_pred, 'region_fn': 0,
            'region_precision': 0.0, 'region_recall': 1.0, 'region_f1': 0.0,
            'total_gt_regions': 0, 'total_pred_regions': n_pred
        }
    
    if n_pred == 0:
        return {
            'region_tp': 0, 'region_fp': 0, 'region_fn': n_gt,
            'region_precision': 1.0, 'region_recall': 0.0, 'region_f1': 0.0,
            'total_gt_regions': n_gt, 'total_pred_regions': 0
        }
    
    # Match predicted regions to GT regions based on IoU
    matched_gt = set()
    matched_pred = set()
    
    for pred_id in range(1, n_pred + 1):
        pred_region = (pred_labeled == pred_id)
        
        best_iou = 0
        best_gt_id = None
        
        for gt_id in range(1, n_gt + 1):
            gt_region = (gt_labeled == gt_id)
            
            intersection = np.sum(pred_region & gt_region)
            union = np.sum(pred_region | gt_region)
            
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold and best_gt_id is not None:
            matched_gt.add(best_gt_id)
            matched_pred.add(pred_id)
    
    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'region_tp': tp,
        'region_fp': fp,
        'region_fn': fn,
        'region_precision': precision,
        'region_recall': recall,
        'region_f1': f1,
        'total_gt_regions': n_gt,
        'total_pred_regions': n_pred
    }

def create_visualization(image_rgb, gt_mask, prob_map, pred_mask, filename, metrics, save_path):
    """Create 4-panel visualization for thesis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(image_rgb)
    ax.set_title('Original Image', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Ground truth overlay
    ax = axes[0, 1]
    ax.imshow(image_rgb)
    if gt_mask is not None:
        gt_overlay = np.zeros((*gt_mask.shape, 4))
        gt_overlay[gt_mask > 0] = [0, 1, 0, 0.5]  # Green overlay
        ax.imshow(gt_overlay)
        ax.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold')
    else:
        ax.set_title('No Annotations', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Probability heatmap
    ax = axes[1, 0]
    im = ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax.set_title('CNN Probability Map', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(Algae)')
    
    # Prediction overlay
    ax = axes[1, 1]
    ax.imshow(image_rgb)
    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[pred_mask > 0] = [1, 0, 0, 0.5]  # Red overlay
    ax.imshow(pred_overlay)
    
    if metrics:
        title_text = f'Predictions\n'
        if 'f1' in metrics:
            title_text += f'Pixel F1: {metrics["f1"]:.3f} | '
        if 'region_f1' in metrics:
            title_text += f'Region F1: {metrics["region_f1"]:.3f}'
        ax.set_title(title_text, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Predictions', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'{filename}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# LOAD ANNOTATIONS
# ============================================================================

print("\n" + "="*70)
print("LOADING ANNOTATIONS")
print("="*70)

with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

# Create mapping from clean filename to image info
image_info_dict = {}
for img in coco_data['images']:
    clean_name = parse_filename(img['file_name'])
    if clean_name not in image_info_dict:
        image_info_dict[clean_name] = {
            'id': img['id'],
            'info': img,
            'json_name': img['file_name']
        }

print(f"   ✓ Loaded {len(coco_data['images'])} annotated images from JSON")
print(f"   ✓ Found {len(coco_data['annotations'])} annotations")

# Get list of all images in directory
all_image_files = sorted(glob(os.path.join(IMAGES_DIR, '*.png')) + 
                         glob(os.path.join(IMAGES_DIR, '*.jpg')))
all_image_filenames = [os.path.basename(f) for f in all_image_files]

annotated_filenames = list(image_info_dict.keys())
unannotated_filenames = [f for f in all_image_filenames if f not in annotated_filenames]

print(f"\n   Images in directory:")
print(f"   • Total: {len(all_image_filenames)}")
print(f"   • Annotated: {len(annotated_filenames)}")
print(f"   • Unannotated: {len(unannotated_filenames)}")

if PROCESS_ALL_IMAGES:
    images_to_process = all_image_filenames
    print(f"\n   ✓ Processing ALL {len(images_to_process)} images")
else:
    images_to_process = annotated_filenames
    print(f"\n   ✓ Processing {len(images_to_process)} annotated images only")

if MAX_IMAGES:
    images_to_process = images_to_process[:MAX_IMAGES]
    print(f"   ⚠️  Limited to first {MAX_IMAGES} images")

# ============================================================================
# PROCESS IMAGES
# ============================================================================

print("\n" + "="*70)
print("PROCESSING IMAGES")
print("="*70)

results_with_gt = []
results_without_gt = []
total_inference_time = 0

for idx, filename in enumerate(tqdm(images_to_process, desc="Processing images")):
    
    # Load image
    image_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"\n   ⚠️  Could not load: {filename}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Check if this image has annotations
    has_annotations = filename in image_info_dict
    
    # Get ground truth mask if available
    gt_mask = None
    if has_annotations:
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        img_id = image_info_dict[filename]['id']
        
        # Get all annotations for this image
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                seg = ann['segmentation']
                
                if isinstance(seg, dict):  # RLE format
                    from pycocotools import mask as mask_util
                    mask = mask_util.decode(seg)
                elif isinstance(seg, list):  # Polygon format
                    for poly in seg:
                        points = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(gt_mask, [points], 1)
                
                # Check category (only mark algae, not non-algae)
                cat_id = ann['category_id']
                cat_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == cat_id)
                if 'non' in cat_name.lower():
                    # This is a non-algae annotation, set to 0
                    if isinstance(seg, dict):
                        from pycocotools import mask as mask_util
                        mask = mask_util.decode(seg)
                        gt_mask[mask > 0] = 0
                    elif isinstance(seg, list):
                        for poly in seg:
                            points = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(gt_mask, [points], 0)
    
    # Run CNN prediction
    prob_map, pred_mask, inference_time = predict_full_image(
        img_rgb, model, PATCH_SIZE, STRIDE, THRESHOLD, DEVICE
    )
    total_inference_time += inference_time
    
    # Compute metrics and save results
    result = {
        'filename': filename,
        'width': w,
        'height': h,
        'inference_time_sec': inference_time,
        'predicted_algae_pixels': np.sum(pred_mask),
        'predicted_algae_percent': 100 * np.sum(pred_mask) / (h * w),
        'mean_probability': np.mean(prob_map),
        'max_probability': np.max(prob_map)
    }
    
    if has_annotations and gt_mask is not None:
        # Pixel-level metrics
        pixel_metrics = compute_pixel_metrics(pred_mask, gt_mask)
        result.update(pixel_metrics)
        
        # Region-level metrics
        region_metrics = compute_region_metrics(pred_mask, gt_mask, iou_threshold=0.3)
        result.update(region_metrics)
        
        results_with_gt.append(result)
        
        # Create visualization
        viz_path = os.path.join(OUTPUT_DIR, f'viz_{filename}')
        create_visualization(img_rgb, gt_mask, prob_map, pred_mask, filename, result, viz_path)
        
    else:
        results_without_gt.append(result)
        
        # Create visualization without GT
        viz_path = os.path.join(OUTPUT_DIR, f'viz_{filename}')
        create_visualization(img_rgb, None, prob_map, pred_mask, filename, None, viz_path)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Statistics for annotated images
if results_with_gt:
    df_with_gt = pd.DataFrame(results_with_gt)
    
    print(f"\n   📊 ANNOTATED IMAGES ({len(results_with_gt)} images):")
    print(f"\n   Pixel-level Performance:")
    print(f"   • Mean F1-Score:  {df_with_gt['f1'].mean():.4f} ± {df_with_gt['f1'].std():.4f}")
    print(f"   • Mean Precision: {df_with_gt['precision'].mean():.4f} ± {df_with_gt['precision'].std():.4f}")
    print(f"   • Mean Recall:    {df_with_gt['recall'].mean():.4f} ± {df_with_gt['recall'].std():.4f}")
    
    print(f"\n   Region-level Performance:")
    print(f"   • Mean F1-Score:  {df_with_gt['region_f1'].mean():.4f} ± {df_with_gt['region_f1'].std():.4f}")
    print(f"   • Mean Precision: {df_with_gt['region_precision'].mean():.4f} ± {df_with_gt['region_precision'].std():.4f}")
    print(f"   • Mean Recall:    {df_with_gt['region_recall'].mean():.4f} ± {df_with_gt['region_recall'].std():.4f}")
    
    print(f"\n   Region Detection Summary:")
    total_gt = df_with_gt['total_gt_regions'].sum()
    total_tp = df_with_gt['region_tp'].sum()
    total_fn = df_with_gt['region_fn'].sum()
    total_fp = df_with_gt['region_fp'].sum()
    
    print(f"   • Total GT regions:      {total_gt}")
    print(f"   • Detected correctly:    {total_tp} ({100*total_tp/total_gt:.1f}%)")
    print(f"   • Missed:                {total_fn} ({100*total_fn/total_gt:.1f}%)")
    print(f"   • False detections:      {total_fp}")
    
    # Save results
    df_with_gt.to_csv(os.path.join(OUTPUT_DIR, 'annotated_results.csv'), index=False)
    print(f"\n   ✓ Saved: annotated_results.csv")

if results_without_gt:
    df_without_gt = pd.DataFrame(results_without_gt)
    
    print(f"\n   📊 UNANNOTATED IMAGES ({len(results_without_gt)} images):")
    print(f"   • Mean predicted algae: {df_without_gt['predicted_algae_percent'].mean():.2f}% ± {df_without_gt['predicted_algae_percent'].std():.2f}%")
    print(f"   • Mean probability: {df_without_gt['mean_probability'].mean():.4f}")
    
    # Save results
    df_without_gt.to_csv(os.path.join(OUTPUT_DIR, 'unannotated_results.csv'), index=False)
    print(f"\n   ✓ Saved: unannotated_results.csv")

# Timing statistics
print(f"\n   ⏱️  COMPUTATIONAL COST:")
print(f"   • Total images processed: {len(images_to_process)}")
print(f"   • Total inference time: {total_inference_time:.2f} seconds")
print(f"   • Average per image: {total_inference_time / len(images_to_process):.3f} seconds")
print(f"   • Images per second: {len(images_to_process) / total_inference_time:.2f}")

# Save timing info for thesis
timing_data = {
    'model': 'CNN (32x32)',
    'total_images': len(images_to_process),
    'total_time_sec': total_inference_time,
    'time_per_image_sec': total_inference_time / len(images_to_process),
    'images_per_sec': len(images_to_process) / total_inference_time,
    'patch_size': PATCH_SIZE,
    'stride': STRIDE,
    'device': str(DEVICE)
}

with open(os.path.join(OUTPUT_DIR, 'timing_info.json'), 'w') as f:
    json.dump(timing_data, f, indent=2)

print(f"\n   ✓ Saved: timing_info.json")

# ============================================================================
# CREATE SUMMARY FIGURE
# ============================================================================

if results_with_gt:
    print("\n" + "="*70)
    print("CREATING SUMMARY FIGURE")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Pixel-level F1 distribution
    ax = axes[0, 0]
    ax.hist(df_with_gt['f1'], bins=15, color='skyblue', edgecolor='darkblue', alpha=0.7)
    ax.axvline(df_with_gt['f1'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df_with_gt["f1"].mean():.3f}')
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Pixel-level F1 Distribution\n(n={len(results_with_gt)} images)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Region-level F1 distribution
    ax = axes[0, 1]
    ax.hist(df_with_gt['region_f1'], bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax.axvline(df_with_gt['region_f1'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df_with_gt["region_f1"].mean():.3f}')
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Region-level F1 Distribution\n(n={len(results_with_gt)} images)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 3. Precision vs Recall scatter
    ax = axes[0, 2]
    ax.scatter(df_with_gt['recall'], df_with_gt['precision'], 
              s=100, alpha=0.6, c='blue', marker='o', label='Pixel-level')
    ax.scatter(df_with_gt['region_recall'], df_with_gt['region_precision'], 
              s=100, alpha=0.6, c='green', marker='^', label='Region-level')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 4. Pixel vs Region metrics comparison
    ax = axes[1, 0]
    metrics = ['Precision', 'Recall', 'F1']
    pixel_vals = [df_with_gt['precision'].mean(), df_with_gt['recall'].mean(), df_with_gt['f1'].mean()]
    region_vals = [df_with_gt['region_precision'].mean(), df_with_gt['region_recall'].mean(), 
                   df_with_gt['region_f1'].mean()]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pixel_vals, width, label='Pixel-level', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, region_vals, width, label='Region-level', color='lightgreen', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Pixel-level vs Region-level Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Detection rate per image
    ax = axes[1, 1]
    detection_rates = df_with_gt['region_tp'] / df_with_gt['total_gt_regions']
    df_sorted = df_with_gt.sort_values('region_f1')
    colors = ['red' if f1 < 0.5 else 'orange' if f1 < 0.7 else 'green' 
              for f1 in df_sorted['region_f1']]
    
    ax.barh(range(len(df_sorted)), df_sorted['region_f1'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in df_sorted['filename']], 
                       fontsize=8)
    ax.set_xlabel('Region-level F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Image Region F1 (sorted)', fontsize=13, fontweight='bold')
    ax.axvline(df_with_gt['region_f1'].mean(), color='blue', linestyle='--', 
              linewidth=2, alpha=0.5, label='Mean')
    ax.set_xlim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='x')
    
    # 6. Inference time distribution
    ax = axes[1, 2]
    ax.hist(df_with_gt['inference_time_sec'], bins=15, color='coral', edgecolor='darkred', alpha=0.7)
    ax.axvline(df_with_gt['inference_time_sec'].mean(), color='blue', linestyle='--', 
              linewidth=2, label=f'Mean: {df_with_gt["inference_time_sec"].mean():.3f}s')
    ax.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Computational Cost per Image', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'CNN Full-Image Performance Summary\n' +
                 f'Threshold={THRESHOLD:.2f} | {len(results_with_gt)} Images | ' +
                 f'Region F1: {df_with_gt["region_f1"].mean():.3f} | ' +
                 f'Pixel F1: {df_with_gt["f1"].mean():.3f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: summary_statistics.png")

# ============================================================================
# COMPLETE
# ============================================================================

print("\n" + "="*70)
print("✅ CNN FULL-IMAGE PREDICTION COMPLETE!")
print("="*70)

print(f"\n📊 Summary:")
print(f"   • Processed: {len(images_to_process)} images")
print(f"   • With annotations: {len(results_with_gt)}")
print(f"   • Without annotations: {len(results_without_gt)}")

if results_with_gt:
    print(f"\n   Performance (annotated images):")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │ PIXEL-LEVEL                         │")
    print(f"   │   F1:        {df_with_gt['f1'].mean():.4f} ± {df_with_gt['f1'].std():.4f}          │")
    print(f"   │   Precision: {df_with_gt['precision'].mean():.4f} ± {df_with_gt['precision'].std():.4f}          │")
    print(f"   │   Recall:    {df_with_gt['recall'].mean():.4f} ± {df_with_gt['recall'].std():.4f}          │")
    print(f"   ├─────────────────────────────────────┤")
    print(f"   │ REGION-LEVEL                        │")
    print(f"   │   F1:        {df_with_gt['region_f1'].mean():.4f} ± {df_with_gt['region_f1'].std():.4f}          │")
    print(f"   │   Precision: {df_with_gt['region_precision'].mean():.4f} ± {df_with_gt['region_precision'].std():.4f}          │")
    print(f"   │   Recall:    {df_with_gt['region_recall'].mean():.4f} ± {df_with_gt['region_recall'].std():.4f}          │")
    print(f"   │   Detection: {100*total_tp/total_gt:.1f}% of algae regions   │")
    print(f"   └─────────────────────────────────────┘")

print(f"\n⏱️  Computational Cost:")
print(f"   • Average inference time: {total_inference_time/len(images_to_process):.3f} sec/image")
print(f"   • Processing rate: {len(images_to_process)/total_inference_time:.2f} images/sec")

print(f"\n📁 Outputs saved to: {OUTPUT_DIR}")
print(f"   • annotated_results.csv - Metrics for annotated images")
print(f"   • unannotated_results.csv - Predictions for unannotated images")
print(f"   • timing_info.json - Computational cost data")
print(f"   • summary_statistics.png - 6-panel summary figure")
print(f"   • viz_*.png - Individual image visualizations")

print("\n🎓 Ready for thesis!")
print("   ✓ Full-image application demonstrated")
print("   ✓ Pixel-level and region-level metrics computed")
print("   ✓ Computational cost measured")
print("   ✓ Visualizations generated")
print("   ✓ CSV data for tables")

print("="*70)