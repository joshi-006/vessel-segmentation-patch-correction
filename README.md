# Vessel Segmentation with Uncertainty-Guided Patch Correction

## 📌 Overview

This project enhances retinal vessel segmentation by introducing a **post-processing patch correction framework** on top of a trained segmentation model.

Rather than modifying the base model, the method focuses on **refining uncertain regions** using uncertainty estimation and a safety-driven update mechanism.

---

## 🧠 Method

### 🔹 Base Model

* FR-UNet ensemble trained on the FIVES dataset

### 🔹 Pipeline

1. **MC Dropout** generates multiple stochastic predictions
2. **Mutual Information (MI)** identifies uncertain regions
3. **Patch Extraction** selects high-uncertainty areas
4. **Patch Correction Model (Attention U-Net)** refines predictions
5. **Safe Update Rule** applies corrections only when confidence improves and uncertainty decreases

> ⚠️ Dice score is used **only for evaluation** and is not involved in any correction decision.

---

## 🚀 Key Idea

> Instead of correcting all regions, the method selectively refines only uncertain patches, while a guarded update mechanism prevents degradation of already correct predictions.

---

## 📊 Results

| Metric      | Value |
| ----------- | ----: |
| Dice Score  | ~0.86 |
| IoU         | ~0.76 |
| Improvement | +0.02 |

### 📈 Additional Insights

* ~97% of images show improvement
* Minimal degradation (~3%)
* Strong gains in challenging regions
* Patch-level Dice improvement > 0.07

---

## 📁 Project Structure

```id="l5lzvb"
project/
│
├── main.py
│   └── End-to-end pipeline orchestration
│
├── training/
│   └── FR-UNet training and optimization
│
├── models/
│   └── Model architectures and loading utilities
│
├── correction/
│   └── Patch-level correction module (core contribution)
│
├── evaluation/
│   └── Metrics computation and performance analysis
│
├── utils/
│   └── Helper functions and shared utilities
│
└── notebooks/
    └── Demo notebook for visualization and experiments
```

---

## ▶️ How to Run

### 1. Install Dependencies

```id="n1"
pip install -r requirements.txt
```

### 2. Download Dataset

FIVES Dataset (Kaggle):
https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation

Place it in:

```id="n2"
data/
└── fundus-image-dataset-for-vessel-segmentation/
```

---

### 3. Download Model Weights

Pretrained FR-UNet ensemble weights are provided separately:

https://drive.google.com/file/d/1HT6GWupH946phBKnBLplz5KhaKGiAr_B/view?usp=sharing

Place them in:

```id="o5p4sp"
models_weights/
├── FRUNet_MC_0.pth
├── FRUNet_MC_1.pth
├── FRUNet_MC_2.pth
├── FRUNet_MC_3.pth
└── FRUNet_MC_4.pth
```

These correspond to multiple trained models used for uncertainty estimation and correction.

---

### 4. Run the Pipeline

```id="n3"
python main.py
```

---

## 🎯 Contributions

* Uncertainty-guided patch selection using Mutual Information
* Safe update mechanism to prevent performance degradation
* Improved segmentation in challenging regions
* Modular and reproducible pipeline

---

## ⚠️ Notes

* Dataset is not included in this repository
* Model weights are provided separately
* Results may vary slightly due to stochasticity in MC Dropout

---

## 📌 Future Work

* Improved uncertainty estimation techniques
* Extension to other medical segmentation tasks
* Optimization for faster inference

---

## 👩‍💻 Authors

* Joshitha
* Vamsika
