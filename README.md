### Usage

Run scripts in order:
```bash
# 1. Preprocessing
python preprocessing/vid2frames.py
python preprocessing/manual-sorting.py
python preprocessing/water_detection-tweaking.py
python preprocessing/img_transformation.py

# 2. Baseline Methods
python simplethreshold/pixel_classification2-COCO.py

# 3. Machine Learning
python RF/RF7-COCO.py
python XGBoost/XGBoost1.py

# 4. Deep Learning (Best Method)
python CNN/CNN3.py
python CNN/CNN-kfold.py
```

---

## Script Documentation

### Preprocessing Scripts

#### `vid2frames.py`
Extracts individual frames from timelapse videos with timestamp-based naming and handles multi-camera processing.

#### `manual-sorting.py`
Interactive tool for manual image quality control, filtering images based on visibility, reflections, and weather conditions.

#### `water_detection-tweaking.py`
Creates water masks through automatic color-based detection combined with interactive manual refinement.

#### `img_transformation.py`
Applies homography transformation using GPS control points to convert angled camera views into top-down orthogonal projections.

#### `figures-preprocessing.py`
Generates publication-quality figures documenting the complete preprocessing pipeline for thesis.

---

### Simple Threshold Methods

#### `pixel_classification2-COCO.py`
Exploratory analysis testing threshold-based classification using color features (RGB, HSV, brightness, ratios) to establish baseline performance.

#### `pixel-classification-results-figure.py`
Generates visual comparison figures showing threshold-based algae detection results on representative images.

---

### Random Forest

#### `RF7-COCO.py`
Trains Random Forest classifier with hyperparameter tuning via GridSearchCV and decision threshold optimization for pixel-level algae detection.

---

### XGBoost

#### `XGBoost1.py`
Trains XGBoost classifier with gradient boosting, handling class imbalance and providing comparable performance to Random Forest with faster training.

---

### CNN (Deep Learning)

#### `CNN3.py`
Main CNN training script with proper image-level data split to prevent leakage, achieving F1-score of 0.975 on unseen test images.

#### `CNN-kfold.py`
5-fold cross-validation providing robust performance estimates with statistical confidence intervals for publication reporting.

#### `CNN-lc.py`
Learning curve analysis determining optimal dataset size by training with varying numbers of labeled images (2-20+).

#### `CNN-pred.py`
Applies trained CNN to complete images using sliding window inference with overlap averaging for smooth spatial predictions.