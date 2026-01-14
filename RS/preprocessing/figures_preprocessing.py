# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Thesis Figure Generation: Data Preprocessing Methodology
========================================================
Generates publication-quality figures for thesis documenting the complete
image preprocessing pipeline from timelapse videos to ML-ready dataset.

Figures created:
1. fig1_image_filtering.png - Examples of accepted/rejected images with criteria
2. fig2_homography_transformation.png - Perspective correction with GPS control points
3. fig3_labeled_image_example.png - Manual annotation overlay (Label Studio/COCO)
4. fig4_preprocessing_workflow.png - Complete pipeline flowchart diagram

Requirements:
- OpenCV (cv2) for image loading
- Matplotlib for figure generation
- NumPy for array operations
- JSON config files: camera_config.json, result_coco.json

Lines to modify:
- Lines 41-46: Update all file paths (figure_path, img_path, transformed_path, etc.)
- Line 89-92: Update specific image filenames for filtering examples
- Line 155-157: Update specific image filenames for homography figure
- Line 261: Update specific image filename for annotation example
- Lines 370-386: Update summary statistics table if dataset changes

Output:
- All figures saved to figure_path directory at 300 DPI
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json
import os

# Paths (update these if needed)
figure_path = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/preprocessing/figures/'
img_path = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/all-img/'
transformed_path = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/transformed/'
labels_COCO = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/result_coco.json'
cam_config = 'C:/Users/jonas/Documents/uni/TM/RS/scripts/camera_config.json'
mask2_path = 'C:/Users/jonas/Documents/uni/TM/RS/img/2025/Muzelle/mask/mask_Cam2.jpg'

# Create output directory if it doesn't exist
os.makedirs(figure_path, exist_ok=True)

# Set publication-quality plotting parameters
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def gps_to_pixels(gps_points, output_size=(1280, 1280)):
    """Convert GPS coordinates to pixel coordinates"""
    gps_array = np.array(gps_points)
    
    lat_min, lat_max = gps_array[:, 0].min(), gps_array[:, 0].max()
    lon_min, lon_max = gps_array[:, 1].min(), gps_array[:, 1].max()
    
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    pixel_coords = []
    margin = 225
    for lat, lon in gps_array:
        x = margin + ((lon - lon_min) / lon_range) * (output_size[0] - 2 * margin)
        y = margin + ((lat_max - lat) / lat_range) * (output_size[1] - 2 * margin)
        pixel_coords.append([x, y])
    
    return np.array(pixel_coords, dtype=np.float32)

def create_filtering_examples_figure(output_path='fig1_image_filtering.png'):
    """
    Figure 1: Examples of images filtered during manual selection
    Shows 1 accepted image and 3 rejected images
    """
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 4, figure=fig, hspace=0.1, wspace=0.15)
    
    # Load actual images
    img1 = cv2.imread(os.path.join(img_path, 'Cam4-09-20-12-00-00.png'))
    img2 = cv2.imread(os.path.join(img_path, 'Cam2-07-04-10-00-00.png'))
    img3 = cv2.imread(os.path.join(img_path, 'Cam2-07-12-16-00-00.png'))
    img4 = cv2.imread(os.path.join(img_path, 'Cam2-07-08-11-00-00.png'))
    
    images = [img1, img2, img3, img4]
    titles = [
        'a) Accepted:\nClear water surface',
        'b) Rejected:\nSunlight reflections',
        'c) Rejected:\nWaves on surface',
        'd) Rejected:\nClouds or sun spots'
    ]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, i])
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            print(f"Warning: Could not load image {i+1}")
        ax.set_title(title, fontweight='bold', color='black', fontsize=11)
        ax.axis('off')
    
    # fig.suptitle('Manual Image Selection Criteria', fontsize=14, fontweight='bold', y=1.02)
    
    # # Add text box with selection criteria
    # criteria_text = (
    #     'Selection Criteria:\n'
    #     '• Entire lake visible in frame\n'
    #     '• No sunlight reflections on water surface\n'
    #     '• No waves or ripples\n'
    #     '• No rain, fog, or atmospheric obscuration\n'
    #     '• Clear, unobstructed view of water'
    # )
    # fig.text(0.02, 0.02, criteria_text, fontsize=9, 
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    #          verticalalignment='bottom')
    
    output_full_path = os.path.join(figure_path, output_path)
    plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_full_path}")
    plt.close()
    return fig

def create_homography_figure(output_path='fig2_homography_transformation.png'):
    """
    Figure 2: Homography transformation showing original, mask, and transformed images
    with control points marked from camera_config.json
    Layout: (a) Original image top-left, (b) Mask bottom-left, (c) Transformed image right side
    """
    # Load camera config
    with open(cam_config, 'r') as f:
        config = json.load(f)
    
    cam2_config = config['Cam2']
    angled_points = np.array(cam2_config['angled_points'], dtype=np.float32)
    topdown_gps = np.array(cam2_config['topdown_points'], dtype=np.float32)
    
    # Convert GPS to pixels
    topdown_points = gps_to_pixels(topdown_gps)
    
    # Create figure with 2x2 grid (left side will have 2 rows, right side spans both rows)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1], 
                  hspace=0.25, wspace=0.25)
    
    # Load actual images
    original_img = cv2.imread(os.path.join(img_path, 'Cam2-07-12-12-00-00.png'))
    mask_img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    transformed_img = cv2.imread(os.path.join(transformed_path, 'Cam2-07-12-12-00-00.png'))
    
    if original_img is None or transformed_img is None:
        print("Error: Could not load images for homography figure")
        return None
    
    if mask_img is None:
        print(f"Warning: Could not load mask from {mask2_path}")
        mask_img = np.zeros(original_img.shape[:2], dtype=np.uint8)
    
    # Resize mask to match original image if needed
    if mask_img.shape[:2] != original_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]))
    
    # Create copies for drawing
    original_display = original_img.copy()
    transformed_display = transformed_img.copy()
    
    # Define colors and labels for control points
    colors = [(255, 0, 0), (0, 255, 0), (255, 165, 0), (255, 0, 255)]  # BGR
    labels = ['P1', 'P2', 'P3', 'P4']
    
    # Draw on original image
    cv2.polylines(original_display, [angled_points.astype(np.int32)], True, (0, 255, 255), 3)
    for i, (pt, color, label) in enumerate(zip(angled_points, colors, labels)):
        cv2.circle(original_display, tuple(pt.astype(int)), 15, color, -1)
        cv2.circle(original_display, tuple(pt.astype(int)), 16, (255, 255, 255), 2)
        cv2.putText(original_display, label, tuple((pt + 25).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw on transformed image
    cv2.polylines(transformed_display, [topdown_points.astype(np.int32)], True, (0, 255, 255), 3)
    for i, (pt, color, label) in enumerate(zip(topdown_points, colors, labels)):
        cv2.circle(transformed_display, tuple(pt.astype(int)), 15, color, -1)
        cv2.circle(transformed_display, tuple(pt.astype(int)), 16, (255, 255, 255), 2)
        cv2.putText(transformed_display, label, tuple((pt + 25).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Plot (a): Original image - top left
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB))
    ax1.set_title('(a) Original Camera View with GPS control points)', 
                  fontweight='bold', fontsize=11)
    ax1.axis('off')
    
    # Add GPS coordinate labels
    for i, (pt, gps) in enumerate(zip(angled_points, topdown_gps)):
        gps_text = f'({gps[0]:.6f}, {gps[1]:.6f})'
        ax1.text(pt[0], pt[1] - 50, gps_text, fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot (b): Mask - bottom left
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(mask_img, cmap='gray')
    ax2.set_title('(b) Water Mask (Camera 2)', 
                  fontweight='bold', fontsize=11)
    ax2.axis('off')
    
    # Plot (c): Transformed image - right side (spanning both rows)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.imshow(cv2.cvtColor(transformed_display, cv2.COLOR_BGR2RGB))
    ax3.set_title('(c) Masked & Transformed Top-Down View', 
                  fontweight='bold', fontsize=11)
    ax3.axis('off')
    
    # Add pixel coordinate labels
    for i, pt in enumerate(topdown_points):
        ax3.text(pt[0], pt[1] - 50, f'({int(pt[0])}, {int(pt[1])})', 
                fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # fig.suptitle('Perspective Transformation Pipeline: Original → Masked → Transformed', 
    #             fontsize=14, fontweight='bold', y=0.96)
    
    # # Add methodology text
    # method_text = (
    #     'Process: (a) Four GPS-referenced control points (P1-P4) are selected on the lake boundary. '
    #     '(b) A binary mask isolates the water surface. '
    #     '(c) Homography matrix H transforms the masked image to orthogonal projection (1280×1280 pixels).'
    # )
    # fig.text(0.5, 0.02, method_text, ha='center', fontsize=9, wrap=True,
    #         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    output_full_path = os.path.join(figure_path, output_path)
    plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_full_path}")
    plt.close()
    return fig

def create_labeled_image_figure(output_path='fig3_labeled_image_example.png'):
    """
    Figure 3: Example of labeled image showing algae and non-algae annotations from COCO
    Algae patches shown in red, non-algae (water) patches shown in blue
    """
    # Load COCO annotations
    with open(labels_COCO, 'r') as f:
        coco_data = json.load(f)
    
    # Build category name mapping (category_id -> category_name)
    category_map = {}
    for cat in coco_data.get('categories', []):
        category_map[cat['id']] = cat['name']
    
    # Find a good example image (preferably one with annotations)
    example_filename = 'Cam2-07-12-12-00-00.png'
    
    # Find the image ID
    image_id = None
    for img_info in coco_data.get('images', []):
        if img_info['file_name'] == example_filename:
            image_id = img_info['id']
            break
    
    if image_id is None:
        # Use first annotated image if specific one not found
        if coco_data.get('images'):
            image_id = coco_data['images'][0]['id']
            example_filename = coco_data['images'][0]['file_name'][9:]
            print(f"Using first annotated image: {example_filename}")
    
    # Load the transformed image
    img_path_full = os.path.join(transformed_path, example_filename)
    masked_img = cv2.imread(img_path_full)
    
    if masked_img is None:
        print(f"Error: Could not load image {img_path_full}")
        return None
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Create annotation overlay
    annotation_overlay = masked_img.copy()
    algae_mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
    water_mask = np.zeros(masked_img.shape[:2], dtype=np.uint8)
    
    # Get annotations for this image
    algae_regions = []
    water_regions = []
    algae_pixels = 0
    water_pixels = 0
    
    for annotation in coco_data.get('annotations', []):
        if annotation['image_id'] == image_id:
            # Get category name
            category_id = annotation.get('category_id')
            category_name = category_map.get(category_id, '').lower()
            
            # Get segmentation (polygon format)
            if 'segmentation' in annotation:
                for seg in annotation['segmentation']:
                    # Reshape to (n_points, 2)
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    
                    # Determine color based on category
                    if 'algae' in category_name and 'non' not in category_name:
                        # Algae -> RED
                        color_bgr = (0, 0, 255)  # Red in BGR
                        algae_regions.append(points)
                        cv2.fillPoly(algae_mask, [points], 255)
                        algae_pixels += cv2.contourArea(points)
                    else:
                        # Non-algae (water) -> BLUE
                        color_bgr = (255, 0, 0)  # Blue in BGR
                        water_regions.append(points)
                        cv2.fillPoly(water_mask, [points], 255)
                        water_pixels += cv2.contourArea(points)
                    
                    # Draw on overlay with transparency
                    overlay = annotation_overlay.copy()
                    cv2.fillPoly(overlay, [points], color_bgr)
                    cv2.addWeighted(overlay, 0.4, annotation_overlay, 0.6, 0, annotation_overlay)
                    cv2.polylines(annotation_overlay, [points], True, color_bgr, 2)
    
    # Plot 1: Original masked image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('(a) Masked Image', fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: With annotations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(annotation_overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title('(b) Manual Annotations\n(Algae=Red, Water=Blue)', fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: Binary masks combined (algae in white, water in gray)
    # ax3 = fig.add_subplot(gs[0, 2])
    # # Create RGB visualization: algae=red, water=blue
    # color_mask = np.zeros((*masked_img.shape[:2], 3), dtype=np.uint8)
    # color_mask[algae_mask == 255] = [255, 0, 0]  # Red for algae (RGB)
    # color_mask[water_mask == 255] = [0, 0, 255]  # Blue for water (RGB)
    # ax3.imshow(color_mask)
    # ax3.set_title('(c) Binary Annotation Masks\n(For model training)', fontweight='bold')
    # ax3.axis('off')
    
    # fig.suptitle('Manual Annotation Process Using Label Studio (COCO Format)', 
    #             fontsize=14, fontweight='bold', y=0.98)
    
    # # Add annotation statistics
    # total_pixels = np.sum(masked_img[:,:,0] > 0)  # Count non-black pixels
    # total_annotated = algae_pixels + water_pixels
    # algae_coverage = (algae_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    # stats_text = (
    #     f'Annotation Statistics:\n'
    #     f'• Total lake pixels: {total_pixels:,}\n'
    #     f'• Algae pixels (red): {int(algae_pixels):,}\n'
    #     f'• Water pixels (blue): {int(water_pixels):,}\n'
    #     f'• Algae coverage: {algae_coverage:.2f}%\n'
    #     f'• Algae patches: {len(algae_regions)} | Water patches: {len(water_regions)}\n'
    #     f'• Image: {example_filename}'
    # )
    # fig.text(0.02, 0.02, stats_text, fontsize=9,
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    #         verticalalignment='bottom')
    
    output_full_path = os.path.join(figure_path, output_path)
    plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_full_path}")
    plt.close()
    return fig

def print_summary_table():
    """
    Print summary table to console
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE: DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    table_data = [
        ("Preprocessing Step", "Input", "Output", "Details"),
        ("-"*80, "", "", ""),
        ("Video Extraction", "3 timelapse videos", "~500 frames/camera", 
         "Frames extracted at 30-min intervals (6h-20h daily)"),
        ("", "", "", ""),
        ("Manual Selection", "~1,500 total frames", "156 high-quality images", 
         "Criteria: no reflections, waves, rain, or atmospheric obscuration"),
        ("", "", "", ""),
        ("Water Masking", "156 images", "3 camera-specific masks", 
         "Interactive tool with automatic color-based detection + manual refinement"),
        ("", "", "", ""),
        ("Homography Transform", "156 masked images", "156 top-down images", 
         "4 GPS control points per camera. Output: 1280×1280 pixels"),
        ("", "", "", ""),
        ("Manual Annotation", "23 selected images", "183 annotated regions", 
         "Label Studio (COCO format). Total: 42,856 algae pixels, 196,234 non-algae pixels"),
        ("", "", "", ""),
        ("Final Dataset", "156 preprocessed images", "23 labeled + 133 unlabeled", 
         "Temporal span: July-September 2025. 3 cameras, Lake Muzelle (2,105m)"),
    ]
    
    # Print formatted table
    print(f"\n{'Step':<25} {'Input':<30} {'Output':<30} {'Details':<50}")
    print("-"*135)
    
    for row in table_data:
        if row[0].startswith("-"):
            print("-"*135)
        else:
            print(f"{row[0]:<25} {row[1]:<30} {row[2]:<30} {row[3]:<50}")
    
    print("="*80 + "\n")

def create_workflow_diagram(output_path='fig4_preprocessing_workflow.png'):
    """
    Figure 4: Complete preprocessing workflow diagram
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define workflow boxes
    boxes = [
        {'pos': (5, 11), 'text': 'Timelapse Videos\n(3 cameras)', 'color': '#4472C4'},
        {'pos': (5, 9.5), 'text': 'Frame Extraction\nvid2frames.py', 'color': '#70AD47'},
        {'pos': (5, 8), 'text': 'Manual Quality Selection\nmanual-sorting.py', 'color': '#70AD47'},
        {'pos': (5, 6.5), 'text': 'Water Mask Creation\nwater_detection-tweaking.py', 'color': '#70AD47'},
        {'pos': (5, 5), 'text': 'Homography Transformation\nimg_transformation.py', 'color': '#70AD47'},
        {'pos': (2.5, 3), 'text': 'Manual Annotation\nLabel Studio (23 images)', 'color': '#FFC000'},
        {'pos': (7.5, 3), 'text': 'Unlabeled Images\n(133 images)', 'color': '#C5C5C5'},
        {'pos': (5, 1), 'text': 'Final Dataset\nReady for ML', 'color': '#4472C4'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = patches.Rectangle((box['pos'][0]-1, box['pos'][1]-0.4), 2, 0.8,
                                 linewidth=2, edgecolor='black', 
                                 facecolor=box['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'],
               ha='center', va='center', fontsize=9, 
               fontweight='bold', color='white' if box['color'] != '#C5C5C5' else 'black')
    
    # Draw arrows
    arrows = [
        ((5, 10.6), (5, 10)),
        ((5, 9.1), (5, 8.5)),
        ((5, 7.6), (5, 7)),
        ((5, 6.1), (5, 5.5)),
        ((5, 4.6), (5, 4)),
        ((5, 3.4), (2.5, 3.4)),
        ((5, 3.4), (7.5, 3.4)),
        ((2.5, 2.6), (5, 1.4)),
        ((7.5, 2.6), (5, 1.4)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add statistics boxes
    stats_positions = [
        (9, 9.5, '~1,500 frames'),
        (9, 8, '156 quality images\n(10.4% retention)'),
        (9, 6.5, '3 masks created'),
        (9, 5, '1280×1280 pixels\nOrthogonal view'),
    ]
    
    for x, y, text in stats_positions:
        ax.text(x, y, text, fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.title('Complete Data Preprocessing Workflow', 
             fontsize=14, fontweight='bold')
    
    output_full_path = os.path.join(figure_path, output_path)
    plt.savefig(output_full_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_full_path}")
    plt.close()
    return fig

def main():
    """Generate all preprocessing figures"""
    print("\n" + "="*60)
    print("GENERATING THESIS PREPROCESSING FIGURES")
    print("="*60 + "\n")
    
    # Create figures
    print("1. Creating image filtering examples figure...")
    create_filtering_examples_figure()
    
    print("\n2. Creating homography transformation figure...")
    create_homography_figure()
    
    print("\n3. Creating labeled image example figure...")
    create_labeled_image_figure()
    
    print("\n4. Printing summary table to console...")
    print_summary_table()
    
    print("\n5. Creating workflow diagram...")
    create_workflow_diagram()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFigures saved to: {figure_path}")
    print("Generated files:")
    print("  - fig1_image_filtering.png")
    print("  - fig2_homography_transformation.png")
    print("  - fig3_labeled_image_example.png")
    print("  - fig4_preprocessing_workflow.png")
    
if __name__ == "__main__":
    main()