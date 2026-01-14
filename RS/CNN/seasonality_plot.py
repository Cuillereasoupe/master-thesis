# -*- coding: utf-8 -*-
"""
Seasonal Algae Coverage Time Series
====================================
Creates a publication-ready time series plot showing algae coverage evolution
over the monitoring season.

Usage:
    python create_seasonality_plot.py

Input files (expected in same directory or update paths below):
    - annotated_results_corrected.csv
    - unannotated_results_corrected.csv

Output:
    - seasonality_plot.png

@author: jonas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
ANNOTATED_CSV = 'annotated_results_corrected.csv'
UNANNOTATED_CSV = 'unannotated_results_corrected.csv'

# Output
OUTPUT_PATH = 'seasonality_plot.png'

# Ground resolution (from thesis)
PIXEL_SIZE_M = 0.21             # 21 cm ground resolution
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2  # 0.0441 m² per pixel

LAKE_PIXELS = 762890            # Total lake pixels from image
YEAR = 2024                     # Year of monitoring (for date parsing)

# Derived lake area
LAKE_AREA_M2 = LAKE_PIXELS * PIXEL_AREA_M2
LAKE_AREA_HA = LAKE_AREA_M2 / 10000

# Figure settings
FIGURE_SIZE = (10, 5)
DPI = 300

print("="*60)
print("AREA CONVERSION (21 cm ground resolution)")
print("="*60)
print(f"Pixel size: {PIXEL_SIZE_M*100:.0f} cm = {PIXEL_SIZE_M} m")
print(f"Pixel area: {PIXEL_AREA_M2:.4f} m²/pixel")
print(f"Lake pixels: {LAKE_PIXELS:,}")
print(f"Implied lake area: {LAKE_AREA_M2:,.0f} m² = {LAKE_AREA_HA:.2f} ha")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename_date(filename, year=YEAR):
    """
    Parse date from filename format: CamX-MM-DD-HH-MM-SS.png
    
    Args:
        filename: e.g., 'Cam2-07-04-13-00-00.png'
        year: Year to use (not in filename)
    
    Returns:
        datetime object
    """
    # Remove extension
    base = filename.replace('.png', '').replace('.jpg', '')
    
    # Split by dash
    parts = base.split('-')
    
    # Expected: ['Cam2', '07', '04', '13', '00', '00']
    if len(parts) >= 6:
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        
        return datetime(year, month, day, hour, minute)
    else:
        raise ValueError(f"Could not parse date from: {filename}")

def pixels_to_area(n_pixels, unit='m2'):
    """
    Convert pixel count to area.
    
    Args:
        n_pixels: Number of pixels
        unit: 'm2' for square meters, 'ha' for hectares
    
    Returns:
        Area in specified unit
    """
    area_m2 = n_pixels * PIXEL_AREA_M2
    
    if unit == 'ha':
        return area_m2 / 10000
    else:
        return area_m2

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def create_seasonality_plot():
    """Generate the seasonal time series plot."""
    
    print("\n" + "="*60)
    print("CREATING SEASONALITY PLOT")
    print("="*60)
    
    # Load data
    print(f"\nLoading data...")
    
    dfs = []
    
    if os.path.exists(ANNOTATED_CSV):
        df_ann = pd.read_csv(ANNOTATED_CSV)
        df_ann['annotated'] = True
        dfs.append(df_ann)
        print(f"  • Annotated: {len(df_ann)} images")
    
    if os.path.exists(UNANNOTATED_CSV):
        df_unann = pd.read_csv(UNANNOTATED_CSV)
        df_unann['annotated'] = False
        dfs.append(df_unann)
        print(f"  • Unannotated: {len(df_unann)} images")
    
    if not dfs:
        raise FileNotFoundError("No CSV files found!")
    
    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"  • Total: {len(df)} images")
    
    # Parse dates
    print(f"\nParsing dates...")
    df['datetime'] = df['filename'].apply(parse_filename_date)
    df = df.sort_values('datetime')
    
    print(f"  • Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
    
    # Calculate areas
    df['algae_area_m2'] = df['predicted_algae_pixels'].apply(lambda x: pixels_to_area(x, 'm2'))
    df['algae_area_ha'] = df['predicted_algae_pixels'].apply(lambda x: pixels_to_area(x, 'ha'))
    df['algae_percent'] = 100 * df['predicted_algae_pixels'] / LAKE_PIXELS
    
    # Statistics
    print(f"\nAlgae coverage statistics:")
    print(f"  • Mean: {df['algae_area_m2'].mean():.1f} m² ({df['algae_percent'].mean():.2f}%)")
    print(f"  • Max: {df['algae_area_m2'].max():.1f} m² ({df['algae_percent'].max():.2f}%)")
    print(f"  • Min: {df['algae_area_m2'].min():.1f} m² ({df['algae_percent'].min():.2f}%)")
    
    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    
    print(f"\nCreating figure...")
    
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE, dpi=150)
    
    # Plot algae area
    ax1.plot(df['datetime'], df['algae_area_m2'], 
             'o-', color='#2E7D32', linewidth=1.5, markersize=6,
             label='Detected algae area')
    
    # Fill under curve
    ax1.fill_between(df['datetime'], df['algae_area_m2'], 
                     alpha=0.3, color='#4CAF50')
    
    # Primary y-axis (m²)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Algae Coverage (m²)', fontsize=11)
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0)
    
    # Secondary y-axis (% of lake)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Lake Coverage (%)', fontsize=11)
    ax2.tick_params(axis='y')
    
    # Sync the two y-axes
    y1_max = ax1.get_ylim()[1]
    y2_max = 100 * y1_max / LAKE_AREA_M2
    ax2.set_ylim(0, y2_max)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Grid
    ax1.grid(True, alpha=0.3, linestyle='-')
    ax1.set_axisbelow(True)
    
    # Title
    ax1.set_title('Seasonal Evolution of Algae Coverage\nLac de la Muzelle (2024)', 
                  fontsize=13, fontweight='medium', pad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    
    # =========================================================================
    # SAVE DATA TABLE
    # =========================================================================
    
    # Save processed data
    output_csv = OUTPUT_PATH.replace('.png', '_data.csv')
    df_export = df[['filename', 'datetime', 'predicted_algae_pixels', 
                    'algae_area_m2', 'algae_area_ha', 'algae_percent']].copy()
    df_export['datetime'] = df_export['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    df_export.to_csv(output_csv, index=False)
    print(f"✓ Saved: {output_csv}")
    
    print("\n" + "="*60)
    print("NOTE: Area calculated using 21 cm ground resolution from thesis.")
    print("="*60)

if __name__ == "__main__":
    create_seasonality_plot()
