# -*- coding: utf-8 -*-
"""
Interactive Image Quality Control Tool
======================================

Controls:
- Y: Keep image and advance
- M: Next image (without keeping)
- N: Previous image
- ESC: Quit

Lines to modify:
- Set source_folder (folder containing images to review)
- Set destination_folder (folder for selected images)
- Set mask_folder (folder where first image per camera is copied)

Output:
- Selected images → destination_folder
- First image per camera → mask_folder (for creating water masks)
"""

import cv2
import os
import shutil
import ctypes

# Paths
source_folder = './data/raw_frames/'
destination_folder = './data/selected/'
mask_folder = './data/masks/'

# Make sure the destination and mask folders exist
os.makedirs(destination_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# Get sorted list of image files
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
image_files.sort()

# Track which cameras already have a "mask" image
camera_mask_added = set()

# Introduction
ctypes.windll.user32.MessageBoxW(
    0,
    "Browse the images (N and M keys) and select (Y key) the good ones based on the following criteria:\n"
    " - Entire lake is visible\n"
    " - No shades on the lake\n"
    " - No reflections on the lake's surface\n"
    " - The water isn't too milky",
    "Info",
    0
)

window_name = "Picture selection (Y=keep, ESC=quit, N=prev, M=next, Z=Undo)"

index = 0
while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(source_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_file}, skipping.")
        index += 1
        continue
    
    # Resize to 33% of original size
    h, w = image.shape[:2]
    image = cv2.resize(image, (w//3, h//3))

    cv2.imshow(window_name, image)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('y'):
        # Always copy to destination folder
        shutil.copy(image_path, os.path.join(destination_folder, image_file))

        # Extract camera ID from 4th character of filename (CamX-...)
        cam_id = image_file[3]  # '1', '2', '3', etc.

        # If this camera has no mask image yet, copy also to mask folder
        if cam_id not in camera_mask_added:
            shutil.copy(image_path, os.path.join(mask_folder, image_file))
            camera_mask_added.add(cam_id)

        index += 1

    elif key == 109:  # M
        if index < len(image_files) - 1:
            index += 1
        else:
            ctypes.windll.user32.MessageBoxW(0, "All images reviewed.", "Info", 0)

    elif key == 110:  # N
        if index > 0:
            index -= 1

    elif key == 27:  # Esc key
        print("Aborted by user.")
        break

cv2.destroyAllWindows()
print("Done processing images.")
