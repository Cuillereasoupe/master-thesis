	# CNN Algae Detection

## Scripts

| Script | Description |
|--------|-------------|
| `CNN4.py` | **Recommended.** CNN training with stratified temporal sampling and early stopping. Prevents overfitting and ensures training images are distributed across the monitoring season. |
| `CNN-kfold.py` | 5-fold cross-validation for robust performance estimates with confidence intervals. |
| `CNN-lc.py` | Learning curve analysis to determine optimal number of training images (2-20+). |
| `CNN-pred.py` | Applies trained model to full images using sliding window inference with overlap averaging. |

## Quick Start
```bash
# Train model (recommended)
python CNN4.py

# Cross-validation for publication metrics
python CNN-kfold.py

# Apply to new images
python CNN-pred.py
```

## Configuration

All scripts expect the following paths to be modified in each script's CONFIGURATION section):
```python
COCO_JSON = './data/result_coco.json'
IMAGES_DIR = './data/transformed/'
OUTPUT_DIR = './output/'
```

## Architecture

- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization + max pooling
- 2 fully connected layers with dropout (0.5)
- Input: 32×32 RGB patches