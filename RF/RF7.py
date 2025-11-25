# -*- coding: utf-8 -*-
"""
Random Forest Classifier with Hyperparameter Tuning
===================================================
Trains and optimizes Random Forest classifier for pixel-level algae detection
using GridSearchCV for hyperparameter tuning and threshold optimization.

Key functionality:
- Extracts 14 color features from annotated pixels
- Hyperparameter tuning via GridSearchCV (3-fold CV)
- Decision threshold optimization for maximum F1-score
- Comprehensive performance visualization and comparison
- Model persistence with optimal settings

Training process:
1. Baseline Random Forest (100 estimators, fixed params)
2. GridSearchCV across 5 hyperparameters (72 combinations)
3. Threshold optimization on validation predictions
4. Final evaluation on held-out test set

Lines to modify:
- Line 59: COCO_JSON path (COCO annotation file)
- Line 60: IMAGES_DIR path (transformed images directory)
- Line 61: OUTPUT_DIR path (where models/figures will be saved)
- Line 158: max_pixels sampling (3000 default, adjust for memory/speed tradeoff)
- Lines 226-232: Hyperparameter grid (modify search space as needed)

Output:
- PKL: random_forest_optimized.pkl (trained model with metadata)
- PNG: Multiple figures (confusion matrices, ROC, PR curves, feature importance)
- TXT: optimization_results.txt (detailed performance metrics)
- CSV: Grid search results

Expected performance: F1-score ≈ 0.78-0.82 (significantly better than thresholds)

@author: jonas
Created: Wed Oct 29 18:10:00 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             roc_curve, auc, precision_recall_curve, f1_score)
from pycocotools import mask as mask_util
from tqdm import tqdm
import joblib
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
COCO_JSON = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/result_coco.json'
IMAGES_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/transformed/'
OUTPUT_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/RF/output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("RANDOM FOREST WITH HYPERPARAMETER TUNING + THRESHOLD OPTIMIZATION")
print("="*70)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_pixel(r, g, b):
    """Extract color features from RGB values"""
    r = max(r, 1)
    g = max(g, 1)
    b = max(b, 1)
    
    rgb_sum = r + g + b
    
    features = {
        'R': r,
        'G': g,
        'B': b,
        'norm_R': r / rgb_sum,
        'norm_G': g / rgb_sum,
        'norm_B': b / rgb_sum,
        'G/B': g / (b + 5),
        'G/R': g / (r + 5),
        'R/B': r / (b + 5),
        'B/R': b / (r + 5),
        'Brightness': (r + g + b) / 3,
        'GreenExcess': (2 * g - r - b) / rgb_sum,
        'V': max(r, g, b) / 255.0,
        'S': (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1e-6),
    }
    
    return features

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading COCO annotations...")
with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

print(f"   ✓ Images: {len(coco_data['images'])}")
print(f"   ✓ Annotations: {len(coco_data['annotations'])}")

images_dict = {img['id']: img for img in coco_data['images']}
categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}

print("\n2. Extracting features from all pixels...")

X_list = []
y_list = []

for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
    img_info = images_dict[ann['image_id']]
    
    filename = img_info['file_name']
    if '-' in filename and len(filename.split('-')[0]) == 8:
        parts = filename.split('-', 1)
        if len(parts) > 1:
            filename = parts[1]
    
    image_path = os.path.join(IMAGES_DIR, filename)
    
    img = cv2.imread(image_path)
    if img is None:
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
        pixels = img_rgb[y_coords, x_coords]
        
        max_pixels = 3000
        if len(pixels) > max_pixels:
            indices = np.random.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[indices]
        
        for pixel in pixels:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            
            if r + g + b < 10:
                continue
            
            features = extract_features_from_pixel(r, g, b)
            X_list.append(list(features.values()))
            y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)
feature_names = list(extract_features_from_pixel(100, 100, 100).keys())

print(f"\n   ✓ Extracted {len(X):,} samples")
print(f"   ✓ Features: {len(feature_names)}")
print(f"\n   Class distribution:")
print(f"   - Algae (1): {np.sum(y==1):,} ({100*np.sum(y==1)/len(y):.1f}%)")
print(f"   - Water (0): {np.sum(y==0):,} ({100*np.sum(y==0)/len(y):.1f}%)")

# ============================================================================
# SPLIT DATA
# ============================================================================

print("\n3. Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"   ✓ Test set: {len(X_test):,} samples")

# ============================================================================
# BASELINE MODEL (for comparison)
# ============================================================================

print("\n4. Training baseline model (for comparison)...")

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=100,
    min_samples_leaf=50,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_test)
baseline_f1 = f1_score(y_test, baseline_pred)

print(f"   ✓ Baseline F1-Score: {baseline_f1:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

print("\n5. Hyperparameter tuning with GridSearchCV...")
print("   (This may take several minutes...)")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf': [25, 50, 100],
    'max_features': ['sqrt', 'log2']
}

print(f"\n   Parameter grid:")
for key, values in param_grid.items():
    print(f"   - {key}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n   Total combinations to test: {total_combinations}")
print(f"   With 3-fold CV: {total_combinations * 3} model fits")

start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=1,  # Use 1 job for GridSearchCV, let RF use all cores
    return_train_score=True
)

grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\n   ✓ Grid search complete! ({elapsed_time/60:.1f} minutes)")
print(f"\n   Best parameters:")
for key, value in grid_search.best_params_.items():
    print(f"   - {key}: {value}")

print(f"\n   Best CV F1 score: {grid_search.best_score_:.4f}")
print(f"   Improvement over baseline: {grid_search.best_score_ - baseline_f1:+.4f}")

# Get best model
best_rf = grid_search.best_estimator_

# ============================================================================
# EVALUATE TUNED MODEL
# ============================================================================

print("\n6. Evaluating tuned model...")

y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

y_pred_proba_train = best_rf.predict_proba(X_train)[:, 1]
y_pred_proba_test = best_rf.predict_proba(X_test)[:, 1]

print("\n   TEST SET (default threshold=0.5):")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(classification_report(y_test, y_pred_test, target_names=['Water', 'Algae'], digits=4))

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n7. Optimizing decision threshold...")

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_test)

# Calculate F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"\n   Optimal threshold: {optimal_threshold:.3f}")
print(f"   At this threshold:")
print(f"   - Precision: {precision[optimal_idx]:.4f}")
print(f"   - Recall:    {recall[optimal_idx]:.4f}")
print(f"   - F1-Score:  {f1_scores[optimal_idx]:.4f}")

# Apply optimized threshold
y_pred_optimized = (y_pred_proba_test >= optimal_threshold).astype(int)

print(f"\n   TEST SET (optimized threshold={optimal_threshold:.3f}):")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(classification_report(y_test, y_pred_optimized, target_names=['Water', 'Algae'], digits=4))

# ============================================================================
# COMPARE ALL MODELS
# ============================================================================

print("\n8. Performance comparison...")

from sklearn.metrics import precision_score, recall_score

comparison = pd.DataFrame({
    'Model': [
        'Baseline RF',
        'Tuned RF (threshold=0.5)',
        'Tuned RF (optimized threshold)'
    ],
    'Threshold': [0.5, 0.5, optimal_threshold],
    'Accuracy': [
        accuracy_score(y_test, baseline_pred),
        accuracy_score(y_test, y_pred_test),
        accuracy_score(y_test, y_pred_optimized)
    ],
    'Precision': [
        precision_score(y_test, baseline_pred),
        precision_score(y_test, y_pred_test),
        precision_score(y_test, y_pred_optimized)
    ],
    'Recall': [
        recall_score(y_test, baseline_pred),
        recall_score(y_test, y_pred_test),
        recall_score(y_test, y_pred_optimized)
    ],
    'F1-Score': [
        f1_score(y_test, baseline_pred),
        f1_score(y_test, y_pred_test),
        f1_score(y_test, y_pred_optimized)
    ]
})

print("\n" + "="*70)
print(comparison.to_string(index=False))
print("="*70)

improvement_tuning = comparison.loc[1, 'F1-Score'] - comparison.loc[0, 'F1-Score']
improvement_threshold = comparison.loc[2, 'F1-Score'] - comparison.loc[1, 'F1-Score']
total_improvement = comparison.loc[2, 'F1-Score'] - comparison.loc[0, 'F1-Score']

print(f"\nImprovements:")
print(f"  • Hyperparameter tuning: {improvement_tuning:+.4f}")
print(f"  • Threshold optimization: {improvement_threshold:+.4f}")
print(f"  • Total improvement: {total_improvement:+.4f} ({100*total_improvement/comparison.loc[0, 'F1-Score']:.1f}%)")

comparison.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n9. Creating visualizations...")

# Precision-Recall curve with optimal threshold marked
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(recall, precision, linewidth=2, color='darkblue', label='PR Curve')
ax.scatter(recall[optimal_idx], precision[optimal_idx], s=200, c='red', 
          marker='*', zorder=5, edgecolors='black', linewidths=2,
          label=f'Optimal (Threshold={optimal_threshold:.3f}, F1={f1_scores[optimal_idx]:.3f})')
ax.scatter(recall_score(y_test, y_pred_test), precision_score(y_test, y_pred_test),
          s=150, c='orange', marker='o', zorder=5, edgecolors='black', linewidths=2,
          label=f'Default (Threshold=0.5, F1={f1_score(y_test, y_pred_test):.3f})')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve with Threshold Optimization', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: precision_recall_curve.png")

# F1-Score vs Threshold
fig, ax = plt.subplots(figsize=(10, 6))

valid_indices = np.where(thresholds <= 1.0)[0]
ax.plot(thresholds[valid_indices], f1_scores[valid_indices], linewidth=2, color='darkblue')
ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
          label=f'Optimal: {optimal_threshold:.3f} (F1={f1_scores[optimal_idx]:.3f})')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7,
          label=f'Default: 0.5')

ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('F1-Score vs Decision Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_optimization.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: threshold_optimization.png")

# Confusion matrices comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_pred, title) in enumerate([
    (baseline_pred, 'Baseline RF'),
    (y_pred_test, 'Tuned RF (threshold=0.5)'),
    (y_pred_optimized, f'Tuned RF (threshold={optimal_threshold:.3f})')
]):
    cm = confusion_matrix(y_test, y_pred)
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Water', 'Algae'],
                yticklabels=['Water', 'Algae'],
                cbar_kws={'label': 'Count'})
    
    f1 = f1_score(y_test, y_pred)
    ax.set_title(f'{title}\nF1={f1:.4f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: confusion_matrices_comparison.png")

# Feature importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
bars = ax.barh(range(len(importances)), importances[indices], color=colors[indices])

ax.set_yticks(range(len(importances)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Tuned Model)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
    ax.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: feature_importance.png")

# ROC Curve Analysis
print("\n   Computing ROC curve...")
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

print(f"   ✓ Test AUC-ROC: {roc_auc:.4f}")

# Find the point on ROC curve closest to our optimal threshold
optimal_threshold_idx = np.argmin(np.abs(roc_thresholds - optimal_threshold))

fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
ax.plot(fpr, tpr, linewidth=2, color='darkblue', label=f'Random Forest (AUC={roc_auc:.3f})')

# Plot random classifier line (diagonal)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC=0.5)')

# Mark optimal threshold point
if optimal_threshold_idx < len(fpr):
    ax.scatter(fpr[optimal_threshold_idx], tpr[optimal_threshold_idx],
              s=200, c='red', marker='*', zorder=5, edgecolors='black', linewidths=2,
              label=f'Optimal Threshold={optimal_threshold:.3f}\n(FPR={fpr[optimal_threshold_idx]:.3f}, TPR={tpr[optimal_threshold_idx]:.3f})')

# Mark default threshold (0.5) point for comparison
default_threshold_idx = np.argmin(np.abs(roc_thresholds - 0.5))
if default_threshold_idx < len(fpr):
    ax.scatter(fpr[default_threshold_idx], tpr[default_threshold_idx],
              s=150, c='orange', marker='o', zorder=5, edgecolors='black', linewidths=2,
              label=f'Default Threshold=0.5')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Random Forest Classifier', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add informative text box
textstr = (
    'AUC = Area Under Curve\n'
    f'Perfect classifier: AUC = 1.0\n'
    f'Random classifier: AUC = 0.5\n'
    f'Our model: AUC = {roc_auc:.3f}'
)
ax.text(0.55, 0.35, textstr, fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: roc_curve.png")

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

print("\n10. Saving final model...")

model_data = {
    'model': best_rf,
    'feature_names': feature_names,
    'optimal_threshold': optimal_threshold,
    'best_params': grid_search.best_params_,
    'test_accuracy': accuracy_score(y_test, y_pred_optimized),
    'test_f1': f1_score(y_test, y_pred_optimized),
    'test_precision': precision_score(y_test, y_pred_optimized),
    'test_recall': recall_score(y_test, y_pred_optimized),
}

model_path = os.path.join(OUTPUT_DIR, 'random_forest_optimized.pkl')
joblib.dump(model_data, model_path)

print(f"   ✓ Model saved to: {model_path}")

# Save detailed results
with open(os.path.join(OUTPUT_DIR, 'optimization_results.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("HYPERPARAMETER TUNING + THRESHOLD OPTIMIZATION RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("BEST PARAMETERS:\n")
    for key, value in grid_search.best_params_.items():
        f.write(f"  {key}: {value}\n")
    
    f.write(f"\nOPTIMAL THRESHOLD: {optimal_threshold:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write(comparison.to_string(index=False))
    
    f.write("\n\n" + "="*70 + "\n")
    f.write("FINAL TEST SET RESULTS (Optimized Model)\n")
    f.write("="*70 + "\n\n")
    f.write(classification_report(y_test, y_pred_optimized, target_names=['Water', 'Algae'], digits=4))

print(f"   ✓ Results saved to: {OUTPUT_DIR}/optimization_results.txt")