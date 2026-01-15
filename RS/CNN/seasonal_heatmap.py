# -*- coding: utf-8 -*-
"""
Create Seasonal Algae Detection Heatmap
=======================================
Generates a publication-ready heatmap figure from aggregated CNN predictions.

Input files (expected in same directory or update paths below):
    - aggregated_frequency_map_corrected.npy  (frequency data)
    - aggregated_total_map.npy                (original lake mask)

Output:
    - seasonal_heatmap.png

@author: jonas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
FREQ_MAP_PATH = 'aggregated_frequency_map.npy'            # Frequency data
LAKE_MASK_PATH = 'aggregated_total_map.npy'               # Original total map for lake boundary

# Output
OUTPUT_PATH = 'seasonal_heatmap.png'

# Figure settings
FIGURE_SIZE = (7, 7)      # Figure size in inches
DPI = 300                  # Output resolution
SMOOTHING_SIGMA = 2        # Gaussian smoothing (0 = no smoothing)
COLORMAP = 'viridis'       # Options: 'viridis', 'plasma', 'YlOrRd', 'RdYlGn_r'
VMIN, VMAX = 0, 1          # Colorbar range

# Title (set to None for no title)
TITLE = ''

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def create_heatmap():
    """Generate the seasonal heatmap figure."""
    
    print("="*60)
    print("CREATING SEASONAL HEATMAP")
    print("="*60)
    
    # Load data
    print(f"\nLoading data...")
    
    if not os.path.exists(FREQ_MAP_PATH):
        raise FileNotFoundError(f"Frequency map not found: {FREQ_MAP_PATH}")
    if not os.path.exists(LAKE_MASK_PATH):
        raise FileNotFoundError(f"Lake mask not found: {LAKE_MASK_PATH}")
    
    freq_map = np.load(FREQ_MAP_PATH)
    total_map = np.load(LAKE_MASK_PATH)
    
    print(f"  • Frequency map shape: {freq_map.shape}")
    print(f"  • Total map shape: {total_map.shape}")
    
    # Create lake mask (any pixel observed at least once)
    lake_mask = total_map > 0
    print(f"  • Lake pixels: {np.sum(lake_mask):,}")
    
    # Apply smoothing if requested
    if SMOOTHING_SIGMA > 0:
        freq_smooth = gaussian_filter(freq_map, sigma=SMOOTHING_SIGMA)
        print(f"  • Applied Gaussian smoothing (σ={SMOOTHING_SIGMA})")
    else:
        freq_smooth = freq_map
    
    # Statistics
    freq_in_lake = freq_map[lake_mask]
    print(f"\nDetection statistics (within lake):")
    print(f"  • Mean frequency: {freq_in_lake.mean():.4f}")
    print(f"  • Max frequency: {freq_in_lake.max():.4f}")
    print(f"  • Pixels with any detection: {np.sum(freq_in_lake > 0):,}")
    print(f"  • Pixels detected >50% of time: {np.sum(freq_in_lake > 0.5):,}")
    
    # Create figure
    print(f"\nCreating figure...")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
    
    # Mask non-lake areas
    freq_masked = np.ma.masked_where(~lake_mask, freq_smooth)
    
    # Set up colormap
    cmap = plt.cm.get_cmap(COLORMAP).copy()
    cmap.set_bad(color='white')  # Color for masked (non-lake) areas
    
    # Plot
    im = ax.imshow(freq_masked, cmap=cmap, vmin=VMIN, vmax=VMAX, 
                   interpolation='bilinear')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
    cbar.set_label('Detection Frequency', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=9)
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Title
    if TITLE:
        ax.set_title(TITLE, fontsize=12, fontweight='medium', pad=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    create_heatmap()
