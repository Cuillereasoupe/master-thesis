# Preprocessing Pipeline

Converts raw timelapse videos into ML-ready orthogonal lake images.

## Pipeline Overview
```
Timelapse Videos → Frame Extraction → Quality Filtering → Water Masking → Homography Transform → Ready for ML
```

## Scripts (run in order)

### 1. `vid2frames.py`
Extracts frames from timelapse videos with timestamp-based naming. Requires adaptations based on the timelapse video.

**Input:** Video files (`.mp4`, `.avi`, etc.)  
**Output:** Individual frames named `CamX-MM-DD-HH-MM-SS.png`

### 2. `manual_sorting.py`
Interactive tool for filtering images based on quality criteria.

**Controls:** Y=keep, M=next, N=previous, ESC=quit

**Note:** Windows-only

### 3. `water_detection_tweaking.py`
Creates water masks through automatic detection + manual refinement.

**Controls:**
- Left click + drag: Add/remove mask areas
- Right click: Toggle add/remove mode
- Arrow keys: Adjust brush size
- S: Save, Q: Quit

### 4. `img_transformation.py`
Applies homography transformation using GPS control points.

**Requires:** `camera_config.json` with control point coordinates per camera.

**Output:** 1280×1280 orthogonal top-down images.

## Configuration

Update paths in each script's configuration section.