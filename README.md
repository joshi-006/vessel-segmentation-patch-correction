# Vessel Segmentation with Uncertainty-Guided Patch Correction

## 🚀 Overview

This project enhances retinal vessel segmentation by introducing a **post-processing patch correction framework** on top of a trained segmentation model.

Rather than modifying the base model, the method focuses on **refining uncertain regions** using epistemic uncertainty estimation and a **safety-driven update mechanism**. 

The pipeline identifies uncertain patches via Mutual Information (MI), corrects them with a dedicated Attention U-Net, and applies corrections only when they improve local model confidence — preventing degradation of already correct areas.

---

## 🎯 Key Features

- Ensemble of 5 FR-UNet models with MC Dropout for robust uncertainty estimation
- Mutual Information (MI) based patch selection targeting epistemically uncertain regions
- Patch-level refinement using Attention U-Net
- **Safe Update Rule**: Accept correction only if mean confidence strictly improves
- Significant improvement in hard and uncertain regions with minimal risk of degradation
- Modular and reproducible codebase

---

## 🧠 Method

### Base Model
- **FR-UNet** (Full Resolution U-Net) ensemble of 5 models trained with different random seeds on the FIVES dataset
- MC Dropout enabled during inference (dropout rate = 0.3)

### Uncertainty Estimation
- Multiple stochastic forward passes (N=30) combining MC Dropout and Test-Time Augmentation
- Computation of **Mutual Information (MI)** to quantify epistemic uncertainty
- High MI regions indicate where the model is most uncertain and likely to make errors

### Patch Correction Pipeline
1. Extract top-K patches with highest MI scores (default: K=16, patch size=81×81)
2. Refine selected patches using a dedicated Attention U-Net correction model
3. **Safe Update Rule**: Apply corrected patch only if `new_confidence > old_confidence + margin`
4. Support for 1–2 iterative correction passes with adaptive stopping

> Note: Dice score is used **only for evaluation**, never for guiding corrections.

---

## 📊 Results

Evaluated on the **test set** of the FIVES dataset.

| Metric                        | Baseline          | Ours              | Improvement          |
|-------------------------------|-------------------|-------------------|----------------------|
| Global Dice Score             | ~0.84             | **~0.86**         | **+0.02**            |
| IoU                           | ~0.74             | **~0.76**         | +0.02                |
| Images Improved               | -                 | **~97%**          | -                    |
| Patch-level Dice Gain         | -                 | **> +0.07**       | Strong targeted gain |
| High-MI / Hard Region Dice    | Lower             | **Significant**   | Clear improvement    |

The method improves segmentation quality in **97% of test images** while rarely degrading performance, thanks to the safe update mechanism.

---

## 📁 Project Structure

```bash
vessel-segmentation-patch-correction/
├── main.py                     # End-to-end inference pipeline
├── requirements.txt
├── README.md
│
├── training/                   # Scripts and configs for training FR-UNet ensemble
├── models/                     # FR-UNet and Attention U-Net model definitions
├── model_weights/              # Pretrained FR-UNet ensemble (5 models)
├── correction/                 # Uncertainty estimation + patch selection + safe correction
├── evaluation/                 # Metrics calculation and qualitative visualization
├── utils/                      # Dataset, preprocessing, and helper functions
├── notebooks/                  # Demo and visualization notebooks
└── results/                    # Generated qualitative results (MI maps, before/after, error maps)

```

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/joshi-006/vessel-segmentation-patch-correction.git
cd vessel-segmentation-patch-correction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the FIVES Dataset

1. Download the dataset from Kaggle:  
   [Fundus Image Dataset for Vessel Segmentation (FIVES)](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation)

2. Extract it and place the folder at the following location:
```bash
./data/fundus-image-dataset-for-vessel-segmentation/
```

   Final expected structure:
```bash
data/
└── fundus-image-dataset-for-vessel-segmentation/
    ├── train/
    │   ├── Original/
    │   └── Ground truth/
    └── test/
        ├── Original/
        └── Ground truth/
```

### 4. Download Pretrained Model Weights

1. Download the 5 FR-UNet ensemble models from Google Drive:  
   [📥 Download Model Weights](https://drive.google.com/file/d/1HT6GWupH946phBKnBLplz5KhaKGiAr_B/view?usp=sharing)

2. Create the folder and place the files:
```bash
mkdir -p model_weights
```

3. Place all downloaded `.pth` files inside `model_weights/`:
```bash
model_weights/
├── FRUNet_MC_0.pth
├── FRUNet_MC_1.pth
├── FRUNet_MC_2.pth
├── FRUNet_MC_3.pth
└── FRUNet_MC_4.pth
```

### 5. Run the Pipeline
```bash
python main.py
```

---

## ⚙️ Hyperparameters (Tuned on Validation Set)

| Parameter                | Value   | Description                              |
|--------------------------|---------|------------------------------------------|
| Patch Size               | 81×81   | Size of extracted patches                |
| Top-K Patches            | 16      | Number of uncertain patches per image    |
| MC Dropout Passes        | 30      | For stable Mutual Information estimation |
| Safe Update Margin       | 0.015   | Minimum confidence improvement required  |
| Correction Passes        | 2       | Aggressive pass + gated refinement pass  |

All hyperparameters were selected **exclusively on the validation split** (15% of training data).

---

## 📸 Qualitative Results

After running the pipeline, the `results/` folder will contain:
- Original vs Ground Truth vs Before vs After comparisons
- Mutual Information uncertainty heatmaps
- False Positive (red) and False Negative (blue) error maps

These visualizations demonstrate effective correction of uncertain vessel boundaries and low-contrast regions.

---

## 📚 Citation

If you use this code or method in your research, please cite:

```bibtex
@misc{joshi2026vesselpatchcorrection,
  author = {Joshitha Namakam and Vamsika},
  title  = {Vessel Segmentation with Uncertainty-Guided Patch Correction},
  year   = {2026},
  url    = {https://github.com/joshi-006/vessel-segmentation-patch-correction}
}
```

---


## 🔮 Future Work

Generalization to other medical segmentation tasks (OCT, MRI, etc.)
Integration of advanced uncertainty methods (evidential learning, Bayesian neural networks)
Optimization for faster inference
Plug-and-play quality control module for clinical pipelines


## 👥 Authors

N.Joshitha
K.Vamsika
