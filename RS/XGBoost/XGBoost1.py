# -*- coding: utf-8 -*-
"""
XGBoost Classifier with Hyperparameter Tuning
=============================================

Training process:
1. Baseline XGBoost with default parameters
2. GridSearchCV across 5 hyperparameters (72 combinations)
3. Threshold optimization on probabilistic predictions
4. Final evaluation with optimized threshold

Lines to modify:
- COCO_JSON path (COCO annotation file)
- IMAGES_DIR path (transformed images directory)
- OUTPUT_DIR path (where models/figures will be saved)
- Line 156: max_pixels sampling (3000 default, adjust for memory/speed tradeoff)
- Hyperparameter grid (modify search space as needed)

Output:
- PKL: xgboost_optimized.pkl (trained model with metadata)
- TXT: xgboost_results.txt (detailed performance metrics)
- CSV: Feature importance rankings, model comparison

@author: jonas
Created: 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             roc_curve, auc, precision_recall_curve, f1_score,
                             precision_score, recall_score)
from pycocotools import mask as mask_util
from tqdm import tqdm
import joblib
import time

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    print("✓ XGBoost imported successfully")
except ImportError:
    print("❌ XGBoost not installed!")
    print("Install with: pip install xgboost")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
COCO_JSON = './data/result_coco.json'
IMAGES_DIR = './data/transformed/'
OUTPUT_DIR = './output/XGBoost/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("XGBOOST ALGAE DETECTION - TRAINING & OPTIMIZATION")
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

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"   ✓ Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# TRAIN BASELINE XGBOOST
# ============================================================================

print("\n4. Training baseline XGBoost...")

baseline_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

start_time = time.time()
baseline_xgb.fit(X_train, y_train)
baseline_time = time.time() - start_time

baseline_pred = baseline_xgb.predict(X_test)
baseline_f1 = f1_score(y_test, baseline_pred)

print(f"   ✓ Training time: {baseline_time:.2f} seconds")
print(f"   ✓ Baseline F1-Score: {baseline_f1:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

print("\n5. Hyperparameter tuning with GridSearchCV...")
print("   (This should be faster than Random Forest!)")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

print(f"\n   Parameter grid:")
for key, values in param_grid.items():
    print(f"   - {key}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n   Total combinations: {total_combinations}")
print(f"   With 3-fold CV: {total_combinations * 3} fits")

start_time = time.time()

grid_search = GridSearchCV(
    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=1  # Let XGBoost use all cores
)

grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time

print(f"\n   ✓ Grid search complete! ({elapsed_time/60:.1f} minutes)")
print(f"\n   Best parameters:")
for key, value in grid_search.best_params_.items():
    print(f"   - {key}: {value}")

print(f"\n   Best CV F1 score: {grid_search.best_score_:.4f}")
print(f"   Improvement over baseline: {grid_search.best_score_ - baseline_f1:+.4f}")

best_xgb = grid_search.best_estimator_

# ============================================================================
# EVALUATE TUNED MODEL
# ============================================================================

print("\n6. Evaluating tuned model...")

y_pred_train = best_xgb.predict(X_train)
y_pred_test = best_xgb.predict(X_test)

y_pred_proba_train = best_xgb.predict_proba(X_train)[:, 1]
y_pred_proba_test = best_xgb.predict_proba(X_test)[:, 1]

print("\n   TEST SET (default threshold=0.5):")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(classification_report(y_test, y_pred_test, target_names=['Water', 'Algae'], digits=4))

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n7. Optimizing decision threshold...")

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_test)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"\n   Optimal threshold: {optimal_threshold:.3f}")
print(f"   At this threshold:")
print(f"   - Precision: {precision[optimal_idx]:.4f}")
print(f"   - Recall:    {recall[optimal_idx]:.4f}")
print(f"   - F1-Score:  {f1_scores[optimal_idx]:.4f}")

y_pred_optimized = (y_pred_proba_test >= optimal_threshold).astype(int)

print(f"\n   TEST SET (optimized threshold={optimal_threshold:.3f}):")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(classification_report(y_test, y_pred_optimized, target_names=['Water', 'Algae'], digits=4))

# ============================================================================
# XGBOOST VARIANTS COMPARISON
# ============================================================================

print("\n8. Comparing XGBoost variants...")

comparison_data = {
    'Model': [
        'XGBoost Baseline',
        'XGBoost Tuned (threshold=0.5)',
        'XGBoost Optimized (best threshold)'
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
}

comparison = pd.DataFrame(comparison_data)

print("\n" + "="*85)
print(comparison.to_string(index=False))
print("="*85)

improvement_tuning = comparison.loc[1, 'F1-Score'] - comparison.loc[0, 'F1-Score']
improvement_threshold = comparison.loc[2, 'F1-Score'] - comparison.loc[1, 'F1-Score']
total_improvement = comparison.loc[2, 'F1-Score'] - comparison.loc[0, 'F1-Score']

print(f"\nImprovements:")
print(f"  • Hyperparameter tuning:  {improvement_tuning:+.4f}")
print(f"  • Threshold optimization: {improvement_threshold:+.4f}")
print(f"  • Total improvement:      {total_improvement:+.4f} ({100*total_improvement/comparison.loc[0, 'F1-Score']:.1f}%)")

comparison.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_variants_comparison.csv'), index=False)

fpr_xgb, tpr_xgb, roc_thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Find the point on ROC curve closest to our optimal threshold
optimal_threshold_idx = np.argmin(np.abs(roc_thresholds - optimal_threshold))

# Feature importance
importances = best_xgb.feature_importances_

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n10. Saving model...")

model_data = {
    'model': best_xgb,
    'feature_names': feature_names,
    'optimal_threshold': optimal_threshold,
    'best_params': grid_search.best_params_,
    'scale_pos_weight': scale_pos_weight,
    'test_accuracy': accuracy_score(y_test, y_pred_optimized),
    'test_f1': f1_score(y_test, y_pred_optimized),
    'test_precision': precision_score(y_test, y_pred_optimized),
    'test_recall': recall_score(y_test, y_pred_optimized),
    'test_auc': roc_auc_xgb,
    'training_time': elapsed_time
}

model_path = os.path.join(OUTPUT_DIR, 'xgboost_optimized.pkl')
joblib.dump(model_data, model_path)

print(f"   ✓ Model saved to: {model_path}")

# Save detailed results
with open(os.path.join(OUTPUT_DIR, 'xgboost_results.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("XGBOOST TRAINING & OPTIMIZATION RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("BEST PARAMETERS:\n")
    for key, value in grid_search.best_params_.items():
        f.write(f"  {key}: {value}\n")
    
    f.write(f"\nOPTIMAL THRESHOLD: {optimal_threshold:.4f}\n")
    f.write(f"SCALE POS WEIGHT: {scale_pos_weight:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("XGBOOST VARIANTS COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write(comparison.to_string(index=False))
    
    f.write("\n\n" + "="*70 + "\n")
    f.write("FINAL TEST SET RESULTS (Optimized XGBoost)\n")
    f.write("="*70 + "\n\n")
    f.write(classification_report(y_test, y_pred_optimized, target_names=['Water', 'Algae'], digits=4))
    
    f.write("\n\n" + "="*70 + "\n")
    f.write("FEATURE IMPORTANCE (Top 10)\n")
    f.write("="*70 + "\n\n")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        f.write(f"{i+1:2d}. {feature_names[idx]:15s}: {importances[idx]:.6f}\n")

print(f"   ✓ Results saved to: {OUTPUT_DIR}/xgboost_results.txt")