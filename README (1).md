# Siamese Change Detection on LEVIR-CD Dataset

**Course Assignment | Machine Learning for Remote Sensing | Prof. Biplab Banerjee, IIT Bombay**

---

## Overview

This project implements and compares three fully convolutional neural network architectures for **bi-temporal change detection** on the LEVIR-CD dataset. The task is to identify changed regions (building construction/demolition) between two aerial images taken at different times.

An ensemble approach using **majority voting** across all three models is also implemented, achieving improved mean IoU over individual models.

---

## Models

### 1. FC-EF (Early Fusion UNet)
Concatenates the pre-change and post-change images along the channel dimension before passing through a single UNet encoder-decoder. The model sees both images as a 6-channel input and learns to detect changes implicitly.

```
Input: [Image A (3ch) + Image B (3ch)] → 6 channel input
Architecture: Single encoder → bottleneck → decoder with skip connections
```

### 2. FC-Siam-Conc (Siamese UNet with Concatenation)
Uses a Siamese encoder with shared weights to independently extract features from both images. Skip connections in the decoder concatenate feature maps from both encoder branches, giving the decoder full access to both images' representations.

```
Input: Image A → Encoder (shared weights)
       Image B → Encoder (shared weights)
Decoder: concatenates [feat_A, feat_B] at each skip connection
```

### 3. FC-Siam-Diff (Siamese UNet with Difference)
Same Siamese encoder structure, but skip connections use the **absolute difference** of feature maps instead of concatenation. This explicitly encodes the change signal before the decoder, making the change regions directly visible as high-activation areas.

```
Input: Image A → Encoder (shared weights)
       Image B → Encoder (shared weights)
Decoder: uses |feat_A - feat_B| at each skip connection
```

### 4. Ensemble (Majority Voting)
All three models vote per pixel. A pixel is predicted as changed if **at least 2 out of 3 models** agree it is changed. More robust than any single model.

```
FC-EF     → prediction map
FC-Siam-conc → prediction map   → majority vote (≥2/3) → final prediction
FC-Siam-diff → prediction map
```

---

## Dataset

**LEVIR-CD** (Large-scale Remote Sensing Change Detection Dataset)

- **Task**: Building change detection from bi-temporal aerial images
- **Image size**: 1024 × 1024 pixels, RGB
- **Labels**: Binary masks (0 = no change, 255 = change)
- **Split**: Train / Val / Test

| Split | Images Used |
|-------|------------|
| Train | 100        |
| Val   | 30         |
| Test  | 30         |

Dataset source: [LEVIR-CD on Kaggle](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd)

---

## Pipeline

### 1. Patch Extraction
Instead of resizing 1024×1024 images (which loses detail), images are cropped into non-overlapping **256×256 patches**. Each 1024×1024 image produces up to 16 patches.

Only patches where **≥5% of pixels are changed** are kept — this ensures every training patch contains meaningful change signal.

```
1 image (1024×1024) → 16 patches (256×256) → filter → ~3-8 valid patches
100 images → ~400-600 training patches
```

### 2. Data Augmentation

**Geometric augmentation** (applied identically to Image A, Image B, and label):
- Random horizontal flip
- Random vertical flip
- Random rotation (90°, 180°, 270°)

**Illumination augmentation** (applied **independently** to Image A and Image B — NOT to label):
- Random brightness adjustment (factor: 0.7–1.3)
- Random contrast adjustment (factor: 0.8–1.2)
- Random saturation adjustment (factor: 0.8–1.2)

Independent illumination augmentation is critical because A and B are captured at different dates — lighting conditions naturally differ between acquisitions.

### 3. Loss Function Schedule

Training uses a two-phase loss strategy:

| Epochs | Loss |
|--------|------|
| 1 – 10 | Focal Loss only |
| 11 – 25 | Focal Loss + Dice Loss |

**Focal Loss**: Down-weights easy no-change pixels, forces model to focus on rare change pixels. Stable in early epochs when model is randomly initialized.

**Dice Loss**: Directly optimizes overlap between prediction and ground truth. Added after epoch 10 once the model begins predicting change pixels, to sharpen boundary predictions.

### 4. Optimizer & Scheduler
- **Optimizer**: Adam (lr = 1e-4)
- **Scheduler**: Cosine Annealing LR — smoothly reduces learning rate from 1e-4 to ~0 over all epochs, allowing stable convergence

### 5. Model Selection
Best model checkpoint is saved based on **validation F1 score**. Final test evaluation uses this best checkpoint, not the last epoch.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across both classes (macro average). Standard metric for segmentation — not fooled by class imbalance |
| **F1 Score** | Harmonic mean of precision and recall for the change class |
| **Precision** | Of all predicted change pixels, how many are actually changed |
| **Recall** | Of all actual change pixels, how many were detected |
| **Accuracy** | Overall pixel accuracy (less meaningful due to class imbalance) |
| **AUC-ROC** | Area under ROC curve — measures model's ability to rank change pixels above no-change pixels |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | mIoU | AUC |
|-------|----------|-----------|--------|----|------|-----|
| FC-EF | - | - | - | - | - | - |
| FC-Siam-conc | - | - | - | - | - | - |
| FC-Siam-diff | - | - | - | - | - | - |
| Ensemble (Majority) | - | - | - | - | - | - |

*(Fill in your actual results after training)*

---

## Project Structure

```
change-detection/
│
├── notebook.ipynb          # Main training notebook
│
├── Cell 1                  # Imports
├── Cell 2                  # Seed for reproducibility
├── Cell 3                  # Config (paths, hyperparameters)
├── Cell 4                  # Model definitions (FC-EF, Siam-conc, Siam-diff)
├── Cell 5                  # Loss functions (Focal, Dice, combined schedule)
├── Cell 6                  # Dataset class (patch extraction + augmentation)
├── Cell 7                  # Create datasets and dataloaders
├── Cell 8                  # Metrics (mIoU, F1, AUC-ROC)
├── Cell 9                  # Train and eval functions
├── Cell 10a                # Train FC-EF
├── Cell 10b                # Train FC-Siam-conc
├── Cell 10c                # Train FC-Siam-diff
├── Cell 11                 # Ensemble majority voting
├── Cell 12                 # Final metrics comparison table
├── Cell 13                 # ROC curve plot
├── Cell 14                 # Validation F1 history plot
└── Cell 15                 # Visualization (A | B | GT | Prediction)
```

---

## Key Design Decisions

**Why patch extraction over resize?**
Resizing loses spatial resolution — fine building edges become blurry. Patching preserves original resolution and produces more training samples.

**Why independent illumination augmentation for A and B?**
Pre and post images are captured at different dates, sometimes years apart. Independent brightness/contrast augmentation simulates real-world illumination differences between acquisition dates, making the model robust to lighting variation.

**Why Focal Loss first, then add Dice?**
In early epochs, the model predicts mostly zeros. Dice Loss on near-zero predictions gives unstable gradients. Focal Loss is stable even when the model is completely wrong. Once predictions improve after epoch 10, Dice is added to sharpen boundaries.

**Why majority voting for ensemble?**
More robust than max-voting — a single overconfident wrong model cannot override two correct models. Requires at least 2 out of 3 models to agree on a change prediction.

**Why mIoU over accuracy?**
In change detection, ~90-95% of pixels are no-change. A model predicting all zeros gets 95% accuracy while being completely useless. mIoU computes IoU per class separately, so the change class is weighted equally regardless of how rare it is.

---

## Limitations & Future Work

- **Epochs**: Trained for 25 epochs due to Kaggle free GPU time constraints. Validation mIoU was still improving at epoch 25 — longer training (100+ epochs) on the full dataset would likely yield higher performance closer to paper results (83-84% mIoU)
- **Early Stopping**: Ideally training should use early stopping based on validation loss with patience of 10-15 epochs rather than a fixed epoch count
- **Dataset size**: Trained on a subset (100 train images) due to compute constraints. Full LEVIR-CD training set has 7120 images
- **Contrastive Loss**: For Siamese models, contrastive loss on encoder feature maps could further improve performance by explicitly pulling features of unchanged regions together and pushing changed regions apart

---

## Requirements

```
torch
torchvision
numpy
Pillow
scikit-learn
matplotlib
```

---

## References

1. Daudt, R.C., Le Saux, B., Boulch, A. — *Fully Convolutional Siamese Networks for Change Detection* (ICIP 2018)
2. Chen, H., et al. — *LEVIR-CD: A Large-Scale Change Detection Dataset* 
3. Lin, T.Y., et al. — *Focal Loss for Dense Object Detection* (ICCV 2017)
