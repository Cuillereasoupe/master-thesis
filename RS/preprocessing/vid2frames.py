# -*- coding: utf-8 -*-
"""
Video Frame Extraction for Timelapse Camera Data
================================================
Extracts individual frames from timelapse videos with timestamp-based naming.
Handles multi-camera processing with flexible timing logic for irregular schedules.

Lines to modify:
- Set your video_folder and output_folder paths
- Define start_times dict with video filenames and timestamps
- (Optional) Define camera_mapping dict for custom camera IDs
- (Optional) List dates with additional 20h images (format: 'YYYY-MM-DD')
"""

import cv2
import os
from datetime import datetime, timedelta
import glob

def save_all_frames(video_path, dir_path, start_time, camera_id, days_with_20h=None, ext='png'):
    """
    Extract all frames from a video and save with timestamps and camera ID
    
    Args:
        video_path: Path to the video file
        dir_path: Directory to save frames
        start_time: Starting timestamp as string 'YYYY-MM-DD HH:MM:SS'
        camera_id: Camera identifier (e.g., 'Cam1', 'Cam2', 'Cam3')
        days_with_20h: List of dates (as strings 'YYYY-MM-DD') that have an additional 20h image
                      If None, assumes no days have 20h images
        ext: File extension for saved frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    os.makedirs(dir_path, exist_ok=True)
    
    frame_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    frame_count = 0
    
    # Convert days_with_20h to a set of date strings for faster lookup
    if days_with_20h is None:
        days_with_20h = set()
    else:
        days_with_20h = set(days_with_20h)
    
    print(f"Processing {camera_id}: {os.path.basename(video_path)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_time.strftime('%m-%d-%H-%M-%S')
        filename = f'{camera_id}-{timestamp}.{ext}'
        
        cv2.imwrite(os.path.join(dir_path, filename), frame)
        print(f"{camera_id} - Frame {frame_count}: {frame_time}")
        
        # Update frame time based on your logic
        current_date = frame_time.strftime('%Y-%m-%d')
        current_hour = frame_time.hour
        
        if current_hour >= 20:
            # Check if current date has 20h image
            if current_date in days_with_20h and current_hour == 19:
                # Next frame will be at 20h on the same day
                frame_time = frame_time + timedelta(hour=1)  # 19h -> 20h
            else:
                # Normal behavior: jump to next day at 6h
                frame_time = frame_time + timedelta(days=1)
                frame_time = frame_time.replace(hour=6)
        else:   
            frame_time = frame_time + timedelta(hour=1)
        
        frame_count += 1
    
    cap.release()
    print(f"Completed {camera_id}: {frame_count} frames extracted")

def process_videos_in_folder(video_folder, output_folder, start_times, camera_mapping=None, days_with_20h_mapping=None):
    """
    Process all videos in a folder
    
    Args:
        video_folder: Path to folder containing videos
        output_folder: Path to folder where frames will be saved
        start_times: Dictionary mapping video filenames to start times
                    OR single start time string if all videos start at the same time
        camera_mapping: Dictionary mapping video filenames to camera IDs
                       If None, will auto-assign Cam1, Cam2, Cam3, etc.
        days_with_20h_mapping: Dictionary mapping video filenames to lists of dates with 20h images
                              OR single list of dates if all videos have the same 20h days
                              OR None if no videos have 20h images
    """
    # Get all video files (common video extensions)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    
    if not video_files:
        print("No video files found in the specified folder!")
        return
    
    # Remove duplicates and sort files for consistent ordering
    video_files = list(set(video_files))
    video_files.sort()
    
    print(f"Found {len(video_files)} video files:")
    for i, video in enumerate(video_files):
        print(f"  {i+1}. {os.path.basename(video)}")
    
    # Process each video
    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        
        # Determine camera ID
        if camera_mapping and video_filename in camera_mapping:
            camera_id = camera_mapping[video_filename]
        else:
            camera_id = f"Cam{i+1}"
        
        # Determine start time
        if isinstance(start_times, dict):
            if video_filename in start_times:
                start_time = start_times[video_filename]    
            else:
                print(f"Warning: No start time specified for {video_filename}, skipping...")
                continue
        else:
            # Single start time for all videos
            start_time = start_times
        
        # Determine days with 20h images
        if days_with_20h_mapping is None:
            days_with_20h = None
        elif isinstance(days_with_20h_mapping, dict):
            days_with_20h = days_with_20h_mapping.get(video_filename, None)
        else:
            # Single list for all videos
            days_with_20h = days_with_20h_mapping
        
        # Process the video
        save_all_frames(video_path, output_folder, start_time, camera_id, days_with_20h)

# Different start times for each video
def example_different_start_times():
    video_folder = './data/videos/'
    output_folder = './data/frames/'
    
    # Map specific video files to their start times
    start_times = {
        'TLC_0007.mp4': '2025-08-07 14:00:00',    }
    
    # Optional: Custom camera mapping
    camera_mapping = {
        'TLC_0007.mp4': 'Cam2',
    }
    
    # Days with additional 20h images
    days_with_20h = []
    
    process_videos_in_folder(video_folder, output_folder, start_times, camera_mapping, days_with_20h)

# Run the processing
if __name__ == "__main__":
    
    # Different start times (comment out the line above and uncomment below)
    example_different_start_times()