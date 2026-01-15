# -*- coding: utf-8 -*-
"""
Interactive Water Mask Creation Tool
====================================

Controls (interactive mode):
- Left click + drag: Add/remove mask areas
- Right click: Toggle between add/remove modes
- Arrow up/down: Adjust brush size
- SPACE: Toggle between overlay and result view
- 'r': Reset to automatic detection
- 's': Save mask
- 'q' or ESC: Quit

Lines to modify:
- Set input_folder path to your mask directory
  
Arguments:
  --auto: Skip interactive mode, use automatic detection only
  --cam-only: Process only files starting with "Cam"

Output:
- Masks saved as: mask_CamX.jpg (where X is camera number from filename)
"""

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
import argparse

class InteractiveWaterDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if image is too large
        height, width = self.original_image.shape[:2]
        
        self.current_image = self.original_image.copy()
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        self.initial_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.mode = 'add'  # 'add' or 'remove'
        self.brush_size = 20
        self.show_result = False  # Toggle between overlay and result
        
        # Mouse callback variables
        self.ix, self.iy = -1, -1
        
        # Create initial automatic detection
        self.auto_detect_water()
        
    def auto_detect_water(self):
        """Automatically detect water using multiple methods and combine them"""
        print("Performing automatic water detection...")
        
        # Method 1: Color-based detection (HSV)
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Create multiple masks for different water types
        masks = []
        
        # Blue water (clear lakes, ocean)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        masks.append(blue_mask)
        
        # Green water (algae-rich)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        masks.append(green_mask)
        
        # Dark water (deep or shadowed)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([179, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        masks.append(dark_mask)
        
        # Method 2: Brightness-based detection
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_mask = 255 - bright_mask  # Invert (water is typically darker)
        
        # Method 3: Edge-based detection (water has fewer edges)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = 255 - edge_mask  # Areas with fewer edges
        
        # Combine all masks
        combined_mask = np.zeros_like(gray)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Add brightness and edge information (weighted)
        combined_mask = cv2.addWeighted(combined_mask, 0.7, bright_mask, 0.2, 0)
        combined_mask = cv2.addWeighted(combined_mask, 0.8, edge_mask, 0.2, 0)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep only the largest connected component (main water body)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if it's large enough to be water
            total_area = combined_mask.shape[0] * combined_mask.shape[1]
            if cv2.contourArea(largest_contour) > total_area * 0.01:  # At least 1% of image
                # Create mask with only the largest contour
                self.mask = np.zeros_like(combined_mask)
                cv2.fillPoly(self.mask, [largest_contour], 255)
            else:
                # If no large water body found, use the combined mask
                self.mask = combined_mask
        else:
            self.mask = combined_mask
            
        # Store initial mask for reset functionality
        self.initial_mask = self.mask.copy()
        
        # Apply initial mask to image
        self.update_display()
        print("Automatic detection complete. Use mouse to refine the mask.")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interactive mask editing"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.draw_on_mask(x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_on_mask(x, y)
            else:
                # Show cursor preview when not drawing
                self.show_cursor_preview(x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click switches mode
            self.mode = 'remove' if self.mode == 'add' else 'add'
            print(f"Switched to {self.mode} mode")
    
    def draw_on_mask(self, x, y):
        """Draw on the mask at the given position"""
        color = 255 if self.mode == 'add' else 0
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
        
        # Real-time preview
        preview = self.get_display_image()
        cursor_color = (0, 255, 0) if self.mode == 'add' else (0, 0, 255)
        cv2.circle(preview, (x, y), self.brush_size, cursor_color, 2)
        cv2.imshow('Interactive Water Detection', preview)
    
    def show_cursor_preview(self, x, y):
        """Show cursor preview when hovering"""
        preview = self.get_display_image()
        cursor_color = (0, 255, 0) if self.mode == 'add' else (0, 0, 255)
        cv2.circle(preview, (x, y), self.brush_size, cursor_color, 1)
        cv2.imshow('Interactive Water Detection', preview)
    
    def get_display_image(self):
        """Get the current display image based on the view mode"""
        if self.show_result:
            # Show masked result
            display = cv2.bitwise_and(self.original_image, self.original_image, mask=self.mask)
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, 'Result View (SPACE to toggle)', (10, 30), font, 0.8, (255, 255, 255), 2)
        else:
            # Show mask overlay with pink color
            display = self.original_image.copy()
            # Create pink overlay where mask is active
            pink_overlay = np.zeros_like(display)
            pink_overlay[self.mask > 0] = [255, 0, 255]  # BGR format - pink/magenta
            # Blend the pink overlay with the original image
            display[self.mask > 0] = cv2.addWeighted(display[self.mask > 0], 0.5, pink_overlay[self.mask > 0], 0.5, 0)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, 'Mask Overlay (SPACE for result)', (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Add mode indicator
        height = display.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        mode_text = f"Mode: {self.mode.upper()} | Brush: {self.brush_size}"
        mode_color = (0, 255, 0) if self.mode == 'add' else (0, 0, 255)
        cv2.putText(display, mode_text, (10, height - 10), font, 0.6, mode_color, 2)
        
        return display
    
    def update_display(self):
        """Update the display with the current mask applied"""
        display = self.get_display_image()
        self.current_image = display
        cv2.imshow('Interactive Water Detection', display)
    
    def run_interactive_session(self):
        """Run the interactive editing session"""
        cv2.namedWindow('Interactive Water Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Interactive Water Detection', self.mouse_callback)
        
        self.update_display()
        
        print("\n=== Interactive Water Detection ===")
        print("Controls:")
        print("- Left click + drag: Add to mask or remove from mask")
        print("- Right click: Switch between add/remove modes")
        print("- Mouse wheel: Change brush size")
        print("- 'r': Reset to automatic detection")
        print("- 's': Save current result")
        print("- 'q' or ESC: Quit")
        print("- SPACE: Toggle between mask overlay and result view")
        print("- Pink areas show detected water")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):  # Reset
                self.mask = self.initial_mask.copy()
                self.update_display()
                print("Reset to automatic detection")
            elif key == ord('s'):  # Save
                self.save_result()
            elif key == ord(' '):  # Toggle view
                self.show_result = not self.show_result
                self.update_display()
                view_name = "Result" if self.show_result else "Mask Overlay"
                print(f"Switched to {view_name} view")
            elif key == 38:  # Arrow Up (in OpenCV keycodes)
                self.brush_size = min(50, self.brush_size + 2)
                print(f"Brush size: {self.brush_size}")
        
            elif key == 40:  # Arrow Down
                self.brush_size = max(5, self.brush_size - 2)
                print(f"Brush size: {self.brush_size}")
        
        cv2.destroyAllWindows()
        return self.mask
    
    def save_result(self):
        """Save the current masked result"""
        # Create output filename
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_dir = os.path.dirname(self.image_path)
        
        # Save mask
        mask_path = os.path.join(output_dir, f"mask_Cam{base_name[3]}.jpg")
        cv2.imwrite(mask_path, self.mask)
        
        print(f"Saved mask to: {mask_path}")

def process_single_image(image_path, interactive=True):
    """Process a single image with optional interactive editing"""
    try:
        detector = InteractiveWaterDetector(image_path)
        
        if interactive:
            final_mask = detector.run_interactive_session()
        else:
            # Just use automatic detection
            final_mask = detector.mask
            detector.save_result()
        
        return final_mask
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_cam_images(folder_path, interactive=True):
    """Process all images starting with 'Cam' in the specified folder"""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid folder")
        return
    
    # Find all image files starting with "Cam"
    cam_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().startswith("cam") and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            cam_files.append(filename)
    
    if not cam_files:
        print(f"No images starting with 'Cam' found in {folder_path}")
        return
    
    print(f"Found {len(cam_files)} images starting with 'Cam':")
    for filename in cam_files:
        print(f"  - {filename}")
    
    # Process each image
    for i, filename in enumerate(cam_files, 1):
        image_path = os.path.join(folder_path, filename)
        print(f"\n=== Processing image {i}/{len(cam_files)}: {filename} ===")
        
        try:
            detector = InteractiveWaterDetector(image_path)
            
            if interactive:
                print(f"Interactive editing for: {filename}")
                print("When you're satisfied with the mask, press 's' to save and 'q' to move to next image")
                final_mask = detector.run_interactive_session()
            else:
                # Just use automatic detection and save
                print(f"Auto-processing: {filename}")
                detector.save_result()
                print(f"Completed auto-processing for: {filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n=== Completed processing all {len(cam_files)} Cam images ===")

def main():
    parser = argparse.ArgumentParser(description='Interactive Water Detection Tool for Cam Images')
    parser.add_argument('path', help='Path to image file or folder')
    parser.add_argument('--auto', action='store_true', help='Use automatic detection only (no interaction)')
    parser.add_argument('--cam-only', action='store_true', help='Process only images starting with "Cam"')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single image
        process_single_image(args.path, interactive=not args.auto)
    elif os.path.isdir(args.path):
        if args.cam_only:
            # Process only Cam images
            process_cam_images(args.path, interactive=not args.auto)
        else:
            # Original behavior - process first image found
            image_files = [f for f in os.listdir(args.path) 
                          if os.path.splitext(f.lower())[1] in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            if image_files:
                first_image = os.path.join(args.path, image_files[0])
                process_single_image(first_image, interactive=not args.auto)
            else:
                print("No image files found in the folder")
    else:
        print(f"Error: {args.path} is not a valid file or folder")

if __name__ == "__main__":
    # Modified example usage for Cam images
    input_folder = "./data/masks/"
    
    # Process all images starting with "Cam" in the folder
    print("Processing all images starting with 'Cam'...")
    process_cam_images(input_folder, interactive=True)