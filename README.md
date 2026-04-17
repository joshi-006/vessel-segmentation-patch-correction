# Vessel Segmentation with Uncertainty-Guided Patch Correction

## 📌 Overview

This project improves retinal vessel segmentation by applying a **post-processing patch correction pipeline** on top of a trained segmentation model.

Instead of modifying the segmentation model itself, the method focuses on **correcting uncertain regions** using uncertainty estimation and a safety-based update rule.

---

## 🧠 Method

### 🔹 Base Model

* FR-UNet ensemble (trained on FIVES dataset)

### 🔹 Key Components

* **MC Dropout** → generates multiple predictions
* **Mutual Information (MI)** → identifies uncertain regions
* **Patch Extraction** → selects high-uncertainty patches
* **Patch Correction Model** (Attention U-Net)
* **Safe Update Rule** → applies correction only if it improves confidence and reduces uncertainty

---

## 🚀 Key Idea

> Instead of correcting all regions, we selectively correct **only uncertain patches**, and apply a **guarded update mechanism** to avoid degrading already correct predictions.

---

## 📊 Results

| Metric      | Value |
| ----------- | ----- |
| Dice Score  | ~0.86 |
| IoU         | ~0.76 |
| Improvement | +0.02 |

### 📈 Additional Insights

* ✅ ~97% images improved
* ⚠️ Minimal degradation (~3%)
* 🔥 Strong improvement in difficult regions
* 🎯 Patch-level Dice improvement > 0.07

---

## 🗂 Project Structure

```text
project/
│
├── main.py                  # Full pipeline (end-to-end)
├── training/                # FR-UNet training
├── correction/              # Patch correction logic (core method)
├── evaluation/              # Metrics and analysis
├── models/                  # Model architectures and loaders
├── utils/                   # Helper functions
├── notebooks/               # Demo notebook
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run pipeline

```bash
python main.py
```

---

## 🎯 Contribution

* Uncertainty-guided patch selection using **Mutual Information**
* Safe update mechanism to prevent degradation
* Significant improvement in **hard regions**
* Modular and reproducible pipeline

---

## 📌 Future Work

* Improve uncertainty estimation methods
* Extend to other medical segmentation tasks
* Optimize correction efficiency

---

## 👩‍💻 Author

Joshitha && Vamsika

---

## ⭐ If you found this useful, consider giving it a star!
