# Brain Tumor Segmentation with U-Net

A deep learning project for medical image segmentation that uses U-Net architecture to identify and segment brain tumors from MRI scans. This project implements advanced image preprocessing techniques and provides a complete pipeline from data loading through model evaluation.

## Features

- **Data Loading & Preprocessing**
  - Automatic image and mask loading from directory structure
  - Image resizing to 128×128 resolution
  - Normalization and binary mask conversion
  - Data validation and error handling

- **Image Enhancement Methods**
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - improves local contrast
  - **Gamma Correction** - adjusts image brightness
  - **Denoising** - three methods: bilateral filter, NLM (Non-Local Means), and Gaussian blur
  - **Laplacian Edge Enhancement** - sharpens tumor boundaries

- **Morphological Operations**
  - Opening (removes small noise)
  - Closing (fills small holes)
  - Elliptical kernel operations

- **Deep Learning Model**
  - U-Net architecture with 4 encoder-decoder levels
  - Skip connections for feature preservation
  - Batch normalization and ReLU activations
  - Sigmoid output for binary segmentation

- **Training & Evaluation**
  - Custom loss functions: Dice loss and BCE-Dice combined loss
  - Multiple metrics: Dice coefficient, IoU (Intersection over Union), Accuracy
  - Image quality metrics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity)
  - Early stopping and model checkpointing
  - Comprehensive visualization of training history

- **Result Visualization**
  - Original image with predicted and ground-truth overlays
  - Prediction comparisons
  - Training/validation loss and metric curves

## Technologies Used

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.7+ |
| **Deep Learning Framework** | TensorFlow 2.10+ |
| **Image Processing** | OpenCV, scikit-image |
| **Scientific Computing** | NumPy, scikit-learn |
| **Data Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |
| **Dataset** | Kaggle Brain Tumor Segmentation |

## Project Structure

```
image-processing-project.ipynb
├── Library Imports & Dependencies
├── Data Loading
│   ├── Load images and masks
│   ├── Data validation
│   └── Data balance analysis
├── Preprocessing
│   ├── Image Enhancement (CLAHE, Gamma, Denoising, Edge Enhancement)
│   └── Morphological Operations (Opening, Closing)
├── Modeling
│   ├── U-Net Architecture Definition
│   ├── Custom Loss Functions
│   ├── Custom Metrics (Dice, IoU, PSNR, SSIM)
│   └── Model Compilation
├── Training
│   ├── Train/Validation Split
│   ├── Callbacks Setup (ModelCheckpoint, EarlyStopping)
│   └── Model Fitting
├── Evaluation
│   ├── Metrics Calculation
│   ├── Predictions on Validation Set
│   └── Result Visualization
└── Analysis & Plotting
    ├── Training history curves
    ├── Result overlays
    └── Multiple prediction samples
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd image-processing-project
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify TensorFlow installation**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Configuration

### Dataset Setup

The project expects the following directory structure:

```
base_dir/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### Configuration Parameters

Update these values in the notebook based on your needs:

```python
# Data loading
base_dir = r"/path/to/brain-tumor-segmentation"
target_size = (128, 128)  # Image resize dimension

# Image enhancement
clahe_clip_limit = 2.0
clahe_tile_grid_size = (8, 8)
gamma = 1.0  # Gamma correction factor

# Model training
learning_rate = 1e-4
batch_size = 16
epochs = 100
validation_split = 0.2  # 20% validation, 80% training

# Thresholding
prediction_threshold = 0.5  # For binary mask conversion
```

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook image-processing-project.ipynb
   ```

2. **Execute cells sequentially:**
   - **Cell 1**: Import all required libraries
   - **Cells 2-5**: Load and inspect dataset
   - **Cells 6-10**: Load all images and masks, check data balance
   - **Cells 12-26**: Apply enhancement and morphological operations
   - **Cells 30-34**: Build U-Net model and compile
   - **Cells 35-36**: Train the model
   - **Cells 37-43**: Evaluate and visualize results

### Typical Workflow

```python
# 1. Load data
images, masks = load_data(base_dir, target_size=(128, 128))

# 2. Check data balance
tumor_ratio = check_data_balance(masks)

# 3. Apply preprocessing (example)
enhanced_image = apply_clahe(images[0])

# 4. Build and compile model
model = unet(input_shape=(128, 128, 1))
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=bce_dice_loss, 
              metrics=[dice_coef_tf, iou_tf])

# 5. Train model
history = model.fit(X_train, Y_train, 
                    validation_data=(X_val, Y_val),
                    epochs=100, batch_size=16, callbacks=callbacks)

# 6. Evaluate on validation set
predictions = model.predict(X_val)
print("Mean Dice:", np.mean(dice_list))
print("Mean IoU:", np.mean(iou_list))

# 7. Visualize predictions
show_overlay(X_val, Y_val, model, idx=0)
```

## How It Works

### 1. Data Loading
- Reads MRI images and corresponding segmentation masks from disk
- Resizes all images to 128×128 pixels for consistent processing
- Normalizes pixel values to [0, 1] range
- Converts masks to binary format (tumor vs. non-tumor)

### 2. Image Enhancement
The project includes four enhancement techniques:

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Improves local contrast in images
- Prevents noise amplification in uniform regions
- Ideal for enhancing subtle medical image features

**Gamma Correction**
- Adjusts image brightness using gamma parameter
- gamma < 1: brightens image
- gamma > 1: darkens image

**Denoising**
- Bilateral Filter: Preserves edges while smoothing
- Non-Local Means (NLM): Effective for additive noise
- Gaussian Blur: Simple blur with fast computation

**Laplacian Edge Enhancement**
- Applies Laplacian operator for edge detection
- Combines original and edge-detected images
- Sharpens tumor boundaries for better segmentation

### 3. Model Architecture

**U-Net Structure:**
```
Input (128×128×1)
    ↓
Encoder Block 1 (64 filters) → skip connection s1
    ↓
Encoder Block 2 (128 filters) → skip connection s2
    ↓
Encoder Block 3 (256 filters) → skip connection s3
    ↓
Encoder Block 4 (512 filters) → skip connection s4
    ↓
Bottleneck (1024 filters)
    ↓
Decoder Block 1 (512 filters) + concatenate(s4)
    ↓
Decoder Block 2 (256 filters) + concatenate(s3)
    ↓
Decoder Block 3 (128 filters) + concatenate(s2)
    ↓
Decoder Block 4 (64 filters) + concatenate(s1)
    ↓
Output Conv + Sigmoid (128×128×1)
```

Each block includes:
- 2× Convolutional layers (3×3 kernels)
- Batch normalization for training stability
- ReLU activation functions

### 4. Loss Function

**BCE-Dice Loss** combines two complementary losses:
```
Total Loss = Binary Cross-Entropy + Dice Loss
```

- **Binary Cross-Entropy**: Penalizes incorrect class predictions
- **Dice Loss**: Focuses on overlap between predicted and ground-truth masks

This combination improves robustness, especially with imbalanced data (more background than tumor).

### 5. Training & Evaluation

**Metrics:**
- **Dice Coefficient**: Measures overlap (range: 0-1, higher is better)
- **IoU (Jaccard Index)**: Intersection over Union (range: 0-1, higher is better)
- **Accuracy**: Pixel-wise correctness
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (range: -1 to 1, higher is better)

**Callbacks:**
- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Stops training if validation loss doesn't improve for 10 epochs

### 6. Output

- Binary segmentation masks (0 = non-tumor, 1 = tumor)
- Predictions saved as `.h5` model file
- Overlaid visualizations showing predicted vs. ground-truth masks

## Examples

### Example 1: Basic Data Loading

```python
# Load images and masks
images, masks = load_data(base_dir="/path/to/dataset", 
                          target_size=(128, 128))
print(f"Loaded {images.shape[0]} images")
print(f"Image shape: {images.shape}")
```

**Output:**
```
Loaded 253 images and 253 masks
Images shape: (253, 128, 128, 1)
Masks shape: (253, 128, 128, 1)
```

### Example 2: Enhance Image with CLAHE

```python
original_image = images[0]
enhanced = apply_clahe(original_image, clip_limit=2.0)

# Compare visually
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title("Original")
plt.figure()
plt.imshow(enhanced.squeeze(), cmap='gray')
plt.title("CLAHE Enhanced")
```

### Example 3: Check Data Balance

```python
tumor_ratio = check_data_balance(masks)
# Output: Tumor pixels: 8.5 (34.2%)
#         Non-tumor pixels: 165.9 (65.8%)
```

### Example 4: Make Prediction

```python
# Predict on a single image
single_image = X_val[0:1]  # (1, 128, 128, 1)
prediction = model.predict(single_image)
binary_prediction = (prediction > 0.5).astype(np.float32)

# Calculate metrics
dice = dice_score(Y_val[0], binary_prediction[0])
iou = iou_score(Y_val[0], binary_prediction[0])
print(f"Dice: {dice:.4f}, IoU: {iou:.4f}")
```

### Example 5: Visualize Results with Overlay

```python
# Show image with predicted tumor overlay
show_overlay(X_val, Y_val, model, idx=5)
# Displays: original MRI | ground-truth overlay | prediction overlay
```

## Troubleshooting

### Issue: `FileNotFoundError: Images folder not found`
**Solution:** Ensure dataset directory structure matches:
```
base_dir/
├── images/
└── masks/
```
Update `base_dir` in the notebook to correct path.

### Issue: Out of Memory (OOM) Error during Training
**Solution:** Reduce batch size in the training cell:
```python
batch_size = 8  # Lower from 16
```
Or reduce image size:
```python
target_size = (64, 64)  # Lower from (128, 128)
```

### Issue: Model Not Improving (Loss Not Decreasing)
**Possible causes and solutions:**
- Learning rate too high/low: Try `1e-5` or `1e-3`
- Insufficient data: Ensure you have enough images
- Data preprocessing issues: Verify image normalization (should be 0-1)
- Imbalanced classes: Use class weights or increase CLAHE strength

### Issue: TensorFlow GPU Not Detected
**Solution:** Verify GPU setup:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
If empty, install GPU drivers and CUDA toolkit for your system.

### Issue: Predictions Are All Black/White (No Gradients)
**Solution:** Check if sigmoid threshold is appropriate:
```python
# Try different thresholds instead of 0.5
predictions_binary = (predictions > 0.3).astype(np.float32)
predictions_binary = (predictions > 0.7).astype(np.float32)
```

### Issue: `No module named 'tensorflow'`
**Solution:** Reinstall TensorFlow:
```bash
pip install --upgrade tensorflow
# Or for GPU support:
pip install tensorflow[and-cuda]
```

## Performance Expectations

On a typical GPU (NVIDIA V100 or better):
- **Training time:** ~15-30 minutes for 100 epochs with 200 images
- **Inference time:** ~50-100ms per image
- **Expected metrics:**
  - Dice Coefficient: 0.85-0.92
  - IoU: 0.75-0.85
  - Accuracy: 95%+

## License

License: Not specified.

---

## Notes

- This is a research/educational implementation designed for Jupyter Notebook environments
- Dataset is sourced from Kaggle's Brain Tumor Segmentation dataset
- Model file is saved as `unet_best.h5` in the working directory
- All image paths in the notebook are configured for Kaggle environment; modify `base_dir` for local use
- GPU acceleration is highly recommended for training; CPU training will be significantly slower
- The notebook processes images sequentially; consider batch processing for production use
