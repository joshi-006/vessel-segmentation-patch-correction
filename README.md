Vessel Segmentation with Uncertainty-Guided Patch Correction

📌 Overview

This project improves retinal vessel segmentation by applying a post-processing patch correction pipeline on top of a trained segmentation model.

Instead of modifying the segmentation model itself, the method focuses on correcting uncertain regions using uncertainty estimation and a safety-based update rule.

---

🧠 Method

🔹 Base Model

- FR-UNet ensemble (trained on FIVES dataset)

🔹 Key Components

- MC Dropout → generates multiple predictions
- Mutual Information (MI) → identifies uncertain regions
- Patch Extraction → selects high-uncertainty patches
- Patch Correction Model (Attention U-Net)
- Safe Update Rule → applies correction only if it improves confidence and reduces uncertainty

---

🚀 Key Idea

«Instead of correcting all regions, we selectively correct only uncertain patches, and apply a guarded update mechanism to avoid degrading already correct predictions.»

---

📊 Results

Metric| Value
Dice Score| ~0.86
IoU| ~0.76
Improvement| +0.02

📈 Additional Insights

- ✅ ~97% images improved
- ⚠️ Minimal degradation (~3%)
- 🔥 Strong improvement in difficult regions
- 🎯 Patch-level Dice improvement > 0.07

---
📁 Project Structure

```
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

▶️ How to Run

1. Install dependencies

pip install -r requirements.txt

2. Download Dataset

This project uses the FIVES dataset from Kaggle:

👉 https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation

After downloading, place it in:

data/
└── fundus-image-dataset-for-vessel-segmentation/

---

3. Download Model Weights

Pretrained FR-UNet ensemble weights are not included due to size constraints.

👉 "Download Weights" (https://drive.google.com/file/d/1HT6GWupH946phBKnBLplz5KhaKGiAr_B/view?usp=sharing)

After downloading:

models_weights/
├── FRUNet_MC_0.pth
├── FRUNet_MC_1.pth
├── FRUNet_MC_2.pth
├── FRUNet_MC_3.pth
├── FRUNet_MC_4.pth

---

4. Run Pipeline

python main.py

---

🎯 Contribution

- Uncertainty-guided patch selection using Mutual Information
- Safe update mechanism to prevent degradation
- Significant improvement in hard regions
- Modular and reproducible pipeline

---

⚠️ Notes

- Dataset is not included in this repository
- Model weights are provided separately
- Results may vary slightly due to randomness in MC Dropout

---

📌 Future Work

- Improve uncertainty estimation methods
- Extend to other medical segmentation tasks
- Optimize correction efficiency

---

👩‍💻 Authors

- Joshitha
- Vamsika

---
