# -*- coding: utf-8 -*-
"""
Threshold Classification Visual Results
========================================
Generates side-by-side visual comparison of threshold-based algae detection
methods on representative lake images. Shows original images alongside
detection results with green overlay for predicted algae regions.

Key functionality:
- Loads sample images showing different algae conditions
- Applies optimal brightness and green channel thresholds
- Creates visual overlays (green = predicted algae)
- Generates publication-quality comparison figure

Lines to modify:
- Line 45: IMAGES_DIR path (transformed images directory)
- Line 46: OUTPUT_DIR path (output directory for figure)
- Lines 50-53: sample_images list (select representative images)
- Line 56: BRIGHTNESS_THRESHOLD (optimal value from analysis, default 95.5)
- Line 57: GREEN_THRESHOLD (optimal value from analysis, default 110)

Output:
- PNG: threshold_visual_results.png (2×3 grid showing 2 images × 3 methods)
- Format: Original | Brightness threshold | Green threshold

Image selection criteria:
- Choose images with varying algae coverage (low, medium, high)
- Select clear images without reflections or artifacts
- Include different lighting conditions if available

F1-scores shown:
- Brightness < 95.5: F1 ≈ 0.598
- Green < 110: F1 ≈ 0.641

@author: jonas
Created: Thu Nov 20 13:06:22 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
IMAGES_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/transformed/'
OUTPUT_DIR = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/simpletreshold/figures/'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'threshold_visual_results.png')

# Select 2 representative images
sample_images = [
    'Cam2-07-12-12-00-00.png',   # Another example
    'Cam4-08-24-14-00-00.png'  # Example with some algae
]

# Optimal thresholds from your analysis
BRIGHTNESS_THRESHOLD = 95.5
GREEN_THRESHOLD = 110

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, img_name in enumerate(sample_images):
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load {img_name}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate brightness (average of RGB)
    brightness = np.mean(img_rgb, axis=2)
    
    # Apply threshold: Brightness < 95.5 → Algae
    algae_mask_brightness = brightness < BRIGHTNESS_THRESHOLD
    
    # Apply Green threshold: Green < 110 → Algae
    green_channel = img_rgb[:, :, 1]
    algae_mask_green = green_channel < GREEN_THRESHOLD
    
    # Create overlays
    overlay_brightness = img_rgb.copy()
    overlay_brightness[algae_mask_brightness] = [0, 255, 0]  # Green
    blended_brightness = cv2.addWeighted(img_rgb, 0.6, overlay_brightness, 0.4, 0)
    
    overlay_green = img_rgb.copy()
    overlay_green[algae_mask_green] = [0, 255, 0]  # Green
    blended_green = cv2.addWeighted(img_rgb, 0.6, overlay_green, 0.4, 0)
    
    # Plot
    row = idx
    
    # Original
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f'Original Image\n{img_name}', fontsize=10)
    axes[row, 0].axis('off')
    
    # Brightness threshold
    axes[row, 1].imshow(blended_brightness)
    axes[row, 1].set_title(f'Brightness < {BRIGHTNESS_THRESHOLD}\nF1={0.598:.3f}', fontsize=10)
    axes[row, 1].axis('off')
    
    # Green threshold
    axes[row, 2].imshow(blended_green)
    axes[row, 2].set_title(f'Green < {GREEN_THRESHOLD}\nF1={0.641:.3f}', fontsize=10)
    axes[row, 2].axis('off')

plt.suptitle('Threshold-Based Classification: Visual Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_FILE}")