# -*- coding: utf-8 -*-
"""
Exploratory Color-Based Algae Detection (Pixel-Level Analysis)
==============================================================
Comprehensive pixel-level analysis of color features for distinguishing algae
from water in lake images. Tests simple threshold-based classification methods
and establishes baseline performance before machine learning approaches.

Key functionality:
- Extracts pixels from COCO-annotated lake images
- Computes 14 color features (RGB, HSV, ratios, brightness, etc.)
- Tests multiple threshold-based classification methods
- Generates ROC curves, confusion matrices, and feature distributions
- Identifies optimal thresholds for single-feature classification

Analysis includes:
- RGB and HSV distributions by class
- Feature importance via AUC scores
- Threshold optimization for brightness (best single feature)
- Comparison of 10+ threshold methods

Lines to modify:
- Line 56: COCO_JSON path (COCO annotation file)
- Line 57: IMAGES_DIR path (transformed/orthorectified images)
- Line 58: OUTPUT_DIR path (where figures/results will be saved)
- Line 142: max_pixels sampling size (5000 default, adjust for memory)

Output:
- CSV: extracted_pixels_coco.csv, threshold_methods_comparison.csv
- PNG: Multiple figures (distributions, ROC curves, confusion matrices)
- Console: Comprehensive analysis summary with optimal thresholds
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from pycocotools import mask as mask_util
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
COCO_JSON = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/result_coco.json'
IMAGES_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/transformed/'
OUTPUT_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/simpletreshold/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("EXPLORATORY COLOR-BASED ALGAE DETECTION (COCO FORMAT)")
print("="*70)

# ============================================================================
# LOAD COCO DATA
# ============================================================================

print("\n1. Loading COCO annotations...")
with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

print(f"   ✓ Images: {len(coco_data['images'])}")
print(f"   ✓ Annotations: {len(coco_data['annotations'])}")
print(f"   ✓ Categories: {coco_data['categories']}")

# Create lookup dictionaries
images_dict = {img['id']: img for img in coco_data['images']}
categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}

print(f"\n   Category mapping:")
for cat_id, cat_name in categories_dict.items():
    print(f"     {cat_id}: {cat_name}")

# ============================================================================
# EXTRACT ALL PIXELS
# ============================================================================

print("\n2. Extracting pixels from all annotations...")

all_pixels = []

for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
    # Get image info
    img_info = images_dict[ann['image_id']]
    
    # Clean filename
    filename = img_info['file_name']
    if '-' in filename and len(filename.split('-')[0]) == 8:
        parts = filename.split('-', 1)
        if len(parts) > 1:
            filename = parts[1]
    
    image_path = os.path.join(IMAGES_DIR, filename)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Decode mask
    seg = ann['segmentation']
    
    if isinstance(seg, dict):
        # RLE format
        mask = mask_util.decode(seg)
    elif isinstance(seg, list):
        # Polygon format
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for poly in seg:
            points = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    else:
        continue
    
    # Resize mask if needed
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Get category name and convert to binary label
    cat_name = categories_dict[ann['category_id']]
    label = 1 if 'algae' in cat_name.lower() and 'non' not in cat_name.lower() else 0
    
    # Extract pixels
    if np.sum(mask) > 0:
        pixels = img_rgb[mask > 0]
        
        # Sample if too many pixels
        max_pixels = 5000
        if len(pixels) > max_pixels:
            indices = np.random.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[indices]
        
        # Add to dataset
        for pixel in pixels:
            all_pixels.append({
                'R': int(pixel[0]),
                'G': int(pixel[1]),
                'B': int(pixel[2]),
                'label': label,
                'label_name': 'Algae' if label == 1 else 'Water',
                'image': filename
            })

df_pixels = pd.DataFrame(all_pixels)

print(f"\n   ✓ Extracted {len(df_pixels):,} pixels")
print(f"   - Algae: {len(df_pixels[df_pixels['label']==1]):,} ({100*len(df_pixels[df_pixels['label']==1])/len(df_pixels):.1f}%)")
print(f"   - Water: {len(df_pixels[df_pixels['label']==0]):,} ({100*len(df_pixels[df_pixels['label']==0])/len(df_pixels):.1f}%)")

# Save
df_pixels.to_csv(os.path.join(OUTPUT_DIR, 'extracted_pixels_coco.csv'), index=False)
print(f"   ✓ Saved to: {OUTPUT_DIR}/extracted_pixels_coco.csv")

# ============================================================================
# COMPUTE COLOR FEATURES (WITH OUTLIER PROTECTION)
# ============================================================================

print("\n3. Computing color features...")

# Check for problematic values
print(f"\n   Checking for edge cases...")
print(f"   - Pixels with R=0: {(df_pixels['R'] == 0).sum():,}")
print(f"   - Pixels with G=0: {(df_pixels['G'] == 0).sum():,}")
print(f"   - Pixels with B=0: {(df_pixels['B'] == 0).sum():,}")
print(f"   - Pixels with B<5: {(df_pixels['B'] < 5).sum():,}")

# Filter out extreme edge cases (pure black or near-black pixels)
# These are likely annotation errors at image borders
print(f"\n   Filtering outliers...")
n_before = len(df_pixels)
df_pixels = df_pixels[(df_pixels['R'] > 0) & (df_pixels['G'] > 0) & (df_pixels['B'] > 0)]
df_pixels = df_pixels[(df_pixels['R'] + df_pixels['G'] + df_pixels['B']) > 10]  # Remove near-black
n_after = len(df_pixels)
print(f"   Removed {n_before - n_after:,} near-black pixels ({100*(n_before-n_after)/n_before:.2f}%)")

# Normalized RGB
rgb_sum = df_pixels['R'] + df_pixels['G'] + df_pixels['B']
df_pixels['norm_R'] = df_pixels['R'] / rgb_sum
df_pixels['norm_G'] = df_pixels['G'] / rgb_sum
df_pixels['norm_B'] = df_pixels['B'] / rgb_sum

# Color ratios - with minimum threshold to avoid division issues
# Add a small constant (5) instead of epsilon to avoid extreme ratios
df_pixels['G/B'] = df_pixels['G'].astype(float) / (df_pixels['B'].astype(float) + 5.0)
df_pixels['G/R'] = df_pixels['G'].astype(float) / (df_pixels['R'].astype(float) + 5.0)
df_pixels['R/B'] = df_pixels['R'].astype(float) / (df_pixels['B'].astype(float) + 5.0)
df_pixels['B/R'] = df_pixels['B'].astype(float) / (df_pixels['R'].astype(float) + 5.0)

# Clip extreme ratios for better visualization
df_pixels['G/B'] = df_pixels['G/B'].clip(0, 5)
df_pixels['G/R'] = df_pixels['G/R'].clip(0, 5)
df_pixels['R/B'] = df_pixels['R/B'].clip(0, 5)
df_pixels['B/R'] = df_pixels['B/R'].clip(0, 5)

# Green excess
df_pixels['GreenExcess'] = (2 * df_pixels['G'] - df_pixels['R'] - df_pixels['B']) / rgb_sum

# Brightness
df_pixels['Brightness'] = (df_pixels['R'] + df_pixels['G'] + df_pixels['B']) / 3.0

# HSV conversion
def rgb_to_hsv_vectorized(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    
    h = np.zeros_like(max_c)
    mask_r = (max_c == r) & (diff > 0)
    mask_g = (max_c == g) & (diff > 0)
    mask_b = (max_c == b) & (diff > 0)
    
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    
    s = np.where(max_c > 0, diff / max_c, 0)
    v = max_c
    
    return h, s, v

h, s, v = rgb_to_hsv_vectorized(df_pixels['R'].values, df_pixels['G'].values, df_pixels['B'].values)
df_pixels['H'] = h
df_pixels['S'] = s
df_pixels['V'] = v

print(f"   ✓ Computed color features")

# ============================================================================
# COLOR STATISTICS
# ============================================================================

print("\n4. Color statistics...")

df_algae = df_pixels[df_pixels['label'] == 1]
df_water = df_pixels[df_pixels['label'] == 0]

print(f"\n   ALGAE ({len(df_algae):,} pixels):")
print(f"   - R: {df_algae['R'].mean():.1f} ± {df_algae['R'].std():.1f}")
print(f"   - G: {df_algae['G'].mean():.1f} ± {df_algae['G'].std():.1f}")
print(f"   - B: {df_algae['B'].mean():.1f} ± {df_algae['B'].std():.1f}")
print(f"   - Brightness: {df_algae['Brightness'].mean():.1f} ± {df_algae['Brightness'].std():.1f}")
print(f"   - G/B: {df_algae['G/B'].mean():.3f} ± {df_algae['G/B'].std():.3f}")
print(f"   - G/R: {df_algae['G/R'].mean():.3f} ± {df_algae['G/R'].std():.3f}")

print(f"\n   WATER ({len(df_water):,} pixels):")
print(f"   - R: {df_water['R'].mean():.1f} ± {df_water['R'].std():.1f}")
print(f"   - G: {df_water['G'].mean():.1f} ± {df_water['G'].std():.1f}")
print(f"   - B: {df_water['B'].mean():.1f} ± {df_water['B'].std():.1f}")
print(f"   - Brightness: {df_water['Brightness'].mean():.1f} ± {df_water['Brightness'].std():.1f}")
print(f"   - G/B: {df_water['G/B'].mean():.3f} ± {df_water['G/B'].std():.3f}")
print(f"   - G/R: {df_water['G/R'].mean():.3f} ± {df_water['G/R'].std():.3f}")

# Check patterns
print(f"\n   KEY DIFFERENCES:")
print(f"   • Green:      Water={df_water['G'].mean():.1f}, Algae={df_algae['G'].mean():.1f}, Diff={df_water['G'].mean()-df_algae['G'].mean():.1f}")
print(f"   • Brightness: Water={df_water['Brightness'].mean():.1f}, Algae={df_algae['Brightness'].mean():.1f}, Diff={df_water['Brightness'].mean()-df_algae['Brightness'].mean():.1f}")
print(f"   • G/B Ratio:  Water={df_water['G/B'].mean():.3f}, Algae={df_algae['G/B'].mean():.3f}, Diff={df_water['G/B'].mean()-df_algae['G/B'].mean():.3f}")
print(f"   • B/R Ratio:  Water={df_water['B/R'].mean():.3f}, Algae={df_algae['B/R'].mean():.3f}, Diff={df_water['B/R'].mean()-df_algae['B/R'].mean():.3f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n5. Creating visualizations...")

# RGB distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, channel in enumerate(['R', 'G', 'B']):
    ax = axes[idx]
    ax.hist(df_water[channel], bins=50, alpha=0.6, label='Water', 
            color='skyblue', density=True, edgecolor='darkblue', linewidth=0.5)
    ax.hist(df_algae[channel], bins=50, alpha=0.6, label='Algae', 
            color='seagreen', density=True, edgecolor='darkgreen', linewidth=0.5)
    ax.axvline(df_water[channel].median(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(df_algae[channel].median(), color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel(f'{channel} Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{channel} Channel\nWater: {df_water[channel].mean():.1f}, Algae: {df_algae[channel].mean():.1f}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('RGB Channel Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rgb_distributions_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: rgb_distributions_coco.png")

# Box plots with better feature selection
features_to_plot = ['R', 'G', 'B', 'Brightness', 'G/B', 'G/R', 'B/R', 'GreenExcess', 'H', 'S', 'V', 'norm_G']

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx]
    data_to_plot = [df_water[feature].dropna(), df_algae[feature].dropna()]
    bp = ax.boxplot(data_to_plot, labels=['Water', 'Algae'], patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('seagreen')
    ax.set_ylabel(feature, fontsize=10)
    ax.set_title(f'{feature}\nW:{df_water[feature].mean():.2f}, A:{df_algae[feature].mean():.2f}', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Color Feature Distributions: Water vs Algae', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_boxplots_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: feature_boxplots_coco.png")

# Scatter plots (sample for performance)
print("   Creating scatter plots (sampling for performance)...")
n_sample = min(10000, len(df_water), len(df_algae))
water_sample = df_water.sample(n=min(n_sample, len(df_water)))
algae_sample = df_algae.sample(n=min(n_sample, len(df_algae)))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.scatter(water_sample['G/B'], water_sample['G'], alpha=0.3, s=1, label='Water', color='skyblue')
ax.scatter(algae_sample['G/B'], algae_sample['G'], alpha=0.3, s=1, label='Algae', color='seagreen')
ax.set_xlabel('Green/Blue Ratio', fontsize=11)
ax.set_ylabel('Green Channel Value', fontsize=11)
ax.set_title('G/B Ratio vs Green Channel', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(water_sample['B'], water_sample['G'], alpha=0.3, s=1, label='Water', color='skyblue')
ax.scatter(algae_sample['B'], algae_sample['G'], alpha=0.3, s=1, label='Algae', color='seagreen')
ax.set_xlabel('Blue Channel Value', fontsize=11)
ax.set_ylabel('Green Channel Value', fontsize=11)
ax.set_title('Green vs Blue', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(water_sample['Brightness'], water_sample['G/B'], alpha=0.3, s=1, label='Water', color='skyblue')
ax.scatter(algae_sample['Brightness'], algae_sample['G/B'], alpha=0.3, s=1, label='Algae', color='seagreen')
ax.set_xlabel('Brightness', fontsize=11)
ax.set_ylabel('G/B Ratio', fontsize=11)
ax.set_title('Brightness vs G/B Ratio', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(water_sample['H'], water_sample['S'], alpha=0.3, s=1, label='Water', color='skyblue')
ax.scatter(algae_sample['H'], algae_sample['S'], alpha=0.3, s=1, label='Algae', color='seagreen')
ax.set_xlabel('Hue', fontsize=11)
ax.set_ylabel('Saturation', fontsize=11)
ax.set_title('Hue vs Saturation', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_plots_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: scatter_plots_coco.png")

# Statistical summary
summary_data = []
for feature in ['R', 'G', 'B', 'Brightness', 'G/B', 'G/R', 'B/R', 'GreenExcess', 'norm_G']:
    summary_data.append({
        'Feature': feature,
        'Water Mean': df_water[feature].mean(),
        'Water Std': df_water[feature].std(),
        'Algae Mean': df_algae[feature].mean(),
        'Algae Std': df_algae[feature].std(),
        'Difference': df_water[feature].mean() - df_algae[feature].mean(),
        'Abs Diff': abs(df_water[feature].mean() - df_algae[feature].mean())
    })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('Abs Diff', ascending=False)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'feature_statistics_coco.csv'), index=False)
print("   ✓ Saved: feature_statistics_coco.csv")

print("\n   Top discriminating features:")
for idx, row in df_summary.head(5).iterrows():
    print(f"   {row['Feature']:15s}: Water={row['Water Mean']:6.2f}, Algae={row['Algae Mean']:6.2f}, Diff={row['Difference']:6.2f}")

# ============================================================================
# THRESHOLD CLASSIFICATION
# ============================================================================

print("\n6. Testing threshold-based classification...")

def test_threshold_method(df, feature, threshold, greater_than=True):
    """Test a simple threshold classification"""
    if greater_than:
        predictions = (df[feature] > threshold).astype(int)
    else:
        predictions = (df[feature] < threshold).astype(int)
    
    # Invert: water=0, algae=1, so if feature > threshold → water (0)
    if greater_than:
        predictions = 1 - predictions
    
    accuracy = accuracy_score(df['label'], predictions)
    
    if len(np.unique(df['label'])) > 1 and len(np.unique(predictions)) > 1:
        cm = confusion_matrix(df['label'], predictions)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Test thresholds
threshold_methods = [
    {'name': 'Brightness < 95', 'feature': 'Brightness', 'threshold': 95, 'greater': False},
    {'name': 'Brightness < 100', 'feature': 'Brightness', 'threshold': 100, 'greater': False},
    {'name': 'Brightness < 105', 'feature': 'Brightness', 'threshold': 105, 'greater': False},
    {'name': 'G < 110', 'feature': 'G', 'threshold': 110, 'greater': False},
    {'name': 'G/B < 1.1', 'feature': 'G/B', 'threshold': 1.1, 'greater': False},
    {'name': 'B/R > 0.9', 'feature': 'B/R', 'threshold': 0.9, 'greater': True},
]

results = []
for method in threshold_methods:
    result = test_threshold_method(df_pixels, method['feature'], 
                                   method['threshold'], method['greater'])
    result['method'] = method['name']
    results.append(result)
    
    print(f"\n   {method['name']}:")
    print(f"   - Accuracy:  {result['accuracy']:.3f}")
    print(f"   - Precision: {result['precision']:.3f}")
    print(f"   - Recall:    {result['recall']:.3f}")
    print(f"   - F1-Score:  {result['f1']:.3f}")

# Find optimal threshold
print("\n7. Finding optimal threshold for Brightness...")

thresholds_to_test = np.linspace(70, 130, 100)
f1_scores = []

for thresh in thresholds_to_test:
    result = test_threshold_method(df_pixels, 'Brightness', thresh, False)
    f1_scores.append(result['f1'])

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_to_test[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"   ✓ Optimal Brightness threshold: {optimal_threshold:.1f}")
print(f"   ✓ F1-Score at optimal: {optimal_f1:.3f}")

# Plot threshold optimization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds_to_test, f1_scores, linewidth=2, color='darkblue')
ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
           label=f'Optimal: {optimal_threshold:.1f} (F1={optimal_f1:.3f})')
ax.set_xlabel('Brightness Threshold', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Threshold Optimization for Brightness\n(Brightness < threshold → Algae)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_optimization_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: threshold_optimization_coco.png")

# ============================================================================
# ADD THIS AFTER STEP 7 (Threshold Optimization)
# ============================================================================

print("\n8. ROC Curve Analysis...")

from sklearn.metrics import roc_curve, auc

# Create figure with 2 subplots for key features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve for Brightness
ax = axes[0]
# Use negative brightness because lower brightness = algae (positive class)
fpr_brightness, tpr_brightness, _ = roc_curve(df_pixels['label'], -df_pixels['Brightness'])
roc_auc_brightness = auc(fpr_brightness, tpr_brightness)

ax.plot(fpr_brightness, tpr_brightness, linewidth=2, color='darkblue',
        label=f'Brightness (AUC = {roc_auc_brightness:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve: Brightness', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# ROC Curve for Green channel
ax = axes[1]
fpr_green, tpr_green, _ = roc_curve(df_pixels['label'], -df_pixels['G'])
roc_auc_green = auc(fpr_green, tpr_green)

ax.plot(fpr_green, tpr_green, linewidth=2, color='darkgreen',
        label=f'Green Channel (AUC = {roc_auc_green:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve: Green Channel', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.suptitle('ROC Curve Analysis for Single-Feature Thresholds', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: roc_curves_coco.png")
print(f"   ✓ Brightness AUC: {roc_auc_brightness:.3f}")
print(f"   ✓ Green Channel AUC: {roc_auc_green:.3f}")

# Compare multiple features in one plot
print("\n   Creating multi-feature ROC comparison...")
features_for_roc = ['Brightness', 'G', 'B', 'G/B', 'B/R']
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['darkblue', 'darkgreen', 'steelblue', 'orange', 'purple']
auc_scores = {}

for feature, color in zip(features_for_roc, colors):
    # Determine if higher or lower values indicate algae
    if feature in ['Brightness', 'G', 'B']:
        # Lower = algae, so use negative
        fpr, tpr, _ = roc_curve(df_pixels['label'], -df_pixels[feature])
    else:
        # For ratios, G/B is higher for algae, B/R is lower for algae
        if feature == 'G/B':
            fpr, tpr, _ = roc_curve(df_pixels['label'], df_pixels[feature])
        else:  # B/R
            fpr, tpr, _ = roc_curve(df_pixels['label'], -df_pixels[feature])
    
    roc_auc = auc(fpr, tpr)
    auc_scores[feature] = roc_auc
    ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{feature} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Comparison of Color Features', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_comparison_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: roc_comparison_coco.png")

print(f"\n   AUC Scores:")
for feature, score in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {feature:15s}: {score:.3f}")

# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

print("\n9. Confusion Matrix for Best Threshold...")

# Generate predictions using optimal threshold
predictions = (df_pixels['Brightness'] < optimal_threshold).astype(int)
cm = confusion_matrix(df_pixels['label'], predictions)

# Create figure with 2 subplots: raw counts and normalized
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Raw counts
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Water', 'Algae'], yticklabels=['Water', 'Algae'])
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title(f'Confusion Matrix - Raw Counts\n(Brightness < {optimal_threshold:.1f})', 
             fontsize=13, fontweight='bold')

# Add text annotations with percentages
tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
ax.text(0.5, 0.25, f'{100*tn/total:.1f}%', ha='center', va='center', fontsize=10, color='gray')
ax.text(1.5, 0.25, f'{100*fp/total:.1f}%', ha='center', va='center', fontsize=10, color='gray')
ax.text(0.5, 1.25, f'{100*fn/total:.1f}%', ha='center', va='center', fontsize=10, color='gray')
ax.text(1.5, 1.25, f'{100*tp/total:.1f}%', ha='center', va='center', fontsize=10, color='gray')

# Plot 2: Normalized (by true class)
ax = axes[1]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Water', 'Algae'], yticklabels=['Water', 'Algae'])
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title(f'Confusion Matrix - Normalized\n(Brightness < {optimal_threshold:.1f})', 
             fontsize=13, fontweight='bold')

plt.suptitle(f'Threshold Classification Performance (F1={optimal_f1:.3f})', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: confusion_matrix_coco.png")

# Print detailed metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / total

print(f"\n   Detailed Performance Metrics:")
print(f"   - True Negatives (Water→Water):  {tn:,} ({100*tn/total:.1f}%)")
print(f"   - False Positives (Water→Algae): {fp:,} ({100*fp/total:.1f}%)")
print(f"   - False Negatives (Algae→Water): {fn:,} ({100*fn/total:.1f}%)")
print(f"   - True Positives (Algae→Algae):  {tp:,} ({100*tp/total:.1f}%)")
print(f"   ---")
print(f"   - Accuracy:    {accuracy:.3f}")
print(f"   - Precision:   {precision:.3f}")
print(f"   - Recall:      {recall:.3f}")
print(f"   - Specificity: {specificity:.3f}")
print(f"   - F1-Score:    {optimal_f1:.3f}")

# ============================================================================
# ADDITIONAL: Compare multiple thresholds
# ============================================================================

print("\n10. Comparing Multiple Threshold Methods...")

# Test best thresholds for different features
best_methods = [
    {'name': 'Brightness < 95.5', 'feature': 'Brightness', 'threshold': optimal_threshold},
    {'name': 'Green < 110', 'feature': 'G', 'threshold': 110},
    {'name': 'Blue < 95', 'feature': 'B', 'threshold': 95},
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, method in enumerate(best_methods):
    predictions = (df_pixels[method['feature']] < method['threshold']).astype(int)
    cm = confusion_matrix(df_pixels['label'], predictions)
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=['Water', 'Algae'], yticklabels=['Water', 'Algae'])
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # Calculate F1
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    ax.set_title(f'{method["name"]}\nF1={f1:.3f}', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrices: Comparison of Single-Feature Thresholds', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices_comparison_coco.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: confusion_matrices_comparison_coco.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n11. Generating Summary Table...")

summary_results = []
for method in threshold_methods:
    result = test_threshold_method(df_pixels, method['feature'], 
                                   method['threshold'], method['greater'])
    summary_results.append({
        'Method': method['name'],
        'Accuracy': result['accuracy'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1-Score': result['f1']
    })

# Add optimal threshold
optimal_result = test_threshold_method(df_pixels, 'Brightness', optimal_threshold, False)
summary_results.append({
    'Method': f'Brightness < {optimal_threshold:.1f} (Optimal)',
    'Accuracy': optimal_result['accuracy'],
    'Precision': optimal_result['precision'],
    'Recall': optimal_result['recall'],
    'F1-Score': optimal_result['f1']
})

df_results = pd.DataFrame(summary_results)
df_results = df_results.sort_values('F1-Score', ascending=False)
df_results.to_csv(os.path.join(OUTPUT_DIR, 'threshold_methods_comparison.csv'), index=False)

print("\n   Threshold Methods Ranked by F1-Score:")
print(df_results.to_string(index=False))
print(f"\n   ✓ Saved: threshold_methods_comparison.csv")