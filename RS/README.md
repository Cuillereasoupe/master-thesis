### Usage

Run scripts in order:
```bash
# 1. Preprocessing
python preprocessing/vid2frames.py
python preprocessing/manual_sorting.py
python preprocessing/water_detection-tweaking.py
python preprocessing/img_transformation.py

# 2. Baseline Methods
python simplethreshold/pixel_classification2-COCO.py

# 3. Machine Learning
python RF/RF7.py
python XGBoost/XGBoost1.py

# 4. Deep Learning
python CNN/CNN4.py
python CNN/CNN-kfold.py
```

---

## Script Documentation

### Preprocessing Scripts

#### `vid2frames.py`
Extracts individual frames from timelapse videos with timestamp-based naming and handles multi-camera processing.

#### `manual_sorting.py`
Interactive tool for manual image quality control, filtering images based on visibility, reflections, and weather conditions.

#### `water_detection-tweaking.py`
Creates water masks through automatic color-based detection combined with interactive manual refinement.

#### `img_transformation.py`
Applies homography transformation using GPS control points to convert angled camera views into top-down orthogonal projections.

---

### Simple Threshold Methods

#### `pixel_classification.py`
Exploratory analysis testing threshold-based classification using color features (RGB, HSV, brightness, ratios) to establish baseline performance.

---

### Random Forest

#### `RF7.py`
Trains Random Forest classifier with hyperparameter tuning via GridSearchCV and decision threshold optimization for pixel-level algae detection.

---

### XGBoost

#### `XGBoost1.py`
Trains XGBoost classifier with gradient boosting, handling class imbalance and providing comparable performance to Random Forest with faster training.

---

### CNN (Deep Learning)

#### `CNN4.py`
CNN training with stratified temporal sampling and early stopping. Prevents overfitting and ensures training images are distributed across the monitoring season.

#### `CNN-kfold.py`
5-fold cross-validation providing robust performance estimates with statistical confidence intervals for publication reporting.

#### `CNN-lc.py`
Learning curve analysis determining optimal dataset size by training with varying numbers of labeled images (2-20+).

#### `CNN-pred.py`
Applies trained CNN to complete images using sliding window inference with overlap averaging for smooth spatial predictions.