# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Image Masking and Homography Transformation
===========================================
Applies water masking and perspective transformation to timelapse images
to generate top-down orthorectified views of alpine lakes.

Key functionality:
- Loads camera-specific masks and transformation parameters from JSON config
- Converts GPS coordinates to pixel coordinates for georeferencing
- Applies homography transformation for perspective correction
- Processes multiple cameras automatically based on filename prefix

Lines to modify:
- Line 126: Set base_path to your project directory
- Line 127: Ensure camera_config.json exists at specified path (see config format in docs)
- Lines 128-131: Set input/output directory paths

Configuration file format (camera_config.json):
{
  "Cam1": {
    "angled_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // 4 corners in source image
    "topdown_points": [[lat1,lon1], [lat2,lon2], [lat3,lon3], [lat4,lon4]]  // GPS coords
  }
}
"""

import os
import cv2
import numpy as np
import json

def gps_to_pixels(gps_points, output_size=(1280, 1280)):
    """
    Convert GPS coordinates to pixel coordinates for top-down view
    
    Args:
        gps_points: Array of [lat, lon] coordinates
        output_size: Desired output image size
    
    Returns:
        Array of [x, y] pixel coordinates
    """
    gps_array = np.array(gps_points)
    
    # Get min/max for normalization
    lat_min, lat_max = gps_array[:, 0].min(), gps_array[:, 0].max()
    lon_min, lon_max = gps_array[:, 1].min(), gps_array[:, 1].max()
    
    # Calculate aspect ratio to maintain proper scale
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    # Convert to pixels with proper scaling
    # Note: latitude decreases as we go down (north to south)
    # longitude increases as we go right (west to east)
    pixel_coords = []
    for lat, lon in gps_array:
        # Normalize and scale to output size with margin
        margin = 225  # pixels from edge
        x = margin + ((lon - lon_min) / lon_range) * (output_size[0] - 2 * margin)
        y = margin + ((lat_max - lat) / lat_range) * (output_size[1] - 2 * margin)
        pixel_coords.append([x, y])
    
    return np.array(pixel_coords, dtype=np.float32)

def load_camera_config(config_path):
    """Load camera configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}

def extract_camera_from_filename(filename):
    """Extract camera identifier from filename (e.g., 'Cam2' from 'Cam2-07-06-12-00-00.png')"""
    return filename.split('-')[0]

def process_image(image_path, mask_path, homography_matrix, output_path, output_masked_path, output_size=(1280, 1280)):
    """
    Apply mask and perspective transformation to an image
    
    Args:
        image_path: Path to input image
        mask_path: Path to mask image
        homography_matrix: 3x3 homography matrix
        output_path: Path to save transformed image
        output_size: Size of output image (width, height)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        return False
    
    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != image.shape[:2]:
        print(f"Resizing mask from {mask.shape[:2]} to {image.shape[:2]}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Threshold mask to ensure binary values (127 is more robust than 254)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply mask to image
    isolated = cv2.bitwise_and(image, image, mask=mask)
    
    # Apply perspective transformation
    transformed = cv2.warpPerspective(isolated, homography_matrix, output_size)
    
    # Save result
    cv2.imwrite(output_path, transformed)
    cv2.imwrite(output_masked_path, isolated)
    print(f"Processed: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
    return True

def main():
    # Paths
    base_path = "C:/Users/jonas/Documents/uni/TM/RS"
    config_path = os.path.join(base_path, "scripts", "camera_config.json")
    img_dir = os.path.join(base_path, "img", "2025", "Muzelle", "img")
    mask_dir = os.path.join(base_path, "img", "2025", "Muzelle", "mask")
    output_dir = os.path.join(base_path, "img", "2025", "Muzelle", "transformed")
    output_masked_dir = os.path.join(base_path, "img", "2025", "Muzelle", "masked_img")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera configuration
    camera_config = load_camera_config(config_path)
    if not camera_config:
        print("Failed to load camera configuration. Exiting.")
        return
    
    # Process each image
    for filename in os.listdir(img_dir):
        if not filename.endswith('.png'):
            continue
        
        # Extract camera identifier
        camera_id = extract_camera_from_filename(filename)
        
        # Check if camera configuration exists
        if camera_id not in camera_config:
            print(f"Warning: No configuration found for {camera_id}. Skipping {filename}")
            continue
        
        # Get camera-specific configuration
        cam_config = camera_config[camera_id]
        angled_points = np.array(cam_config['angled_points'], dtype=np.float32)
        topdown_gps = cam_config['topdown_points']
        
        # Convert GPS coordinates to pixel coordinates
        topdown_points = gps_to_pixels(topdown_gps)
        
        # Compute homography matrix
        homography_matrix, status = cv2.findHomography(angled_points, topdown_points)
        
        # Define paths
        image_path = os.path.join(img_dir, filename)
        output_path = os.path.join(output_dir, filename)
        output_masked_path = os.path.join(output_masked_dir, filename)
        
        # Try to find mask with different extensions
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_mask = os.path.join(mask_dir, f"mask_{camera_id}{ext}")
            if os.path.exists(potential_mask):
                mask_path = potential_mask
                break
        
        # Check if mask exists
        if mask_path is None:
            print(f"Warning: Mask not found for {camera_id}. Skipping {filename}")
            continue
        
        # Process image
        process_image(image_path, mask_path, homography_matrix, output_path, output_masked_path)
    
    print(f"\nProcessing complete! Transformed images saved to: {output_dir}")

if __name__ == "__main__":
    main()