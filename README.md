# 3D CoAt U SegNet

A lightweight and accurate 3D deep learning framework for ischemic stroke lesion segmentation from Non-Contrast CT (NCCT) scans.

## 🧠 Model Overview

This repository implements **3D CoAt U SegNet**, which combines:
- MBConv blocks for efficient local feature extraction
- Transformer-based self-attention for global context modeling
- Multi-Level Dilated Residual (MLDR) blocks for decoding

> Designed specifically for the segmentation of **hypo-intense ischemic stroke lesions** in NCCT brain volumes.

---

## 📂 Directory Structure
```
├── main.py            # Entry point
├── train.py           # Training logic
├── models/
│   └── model.py       # CoAt U SegNet architecture
├── utils/
│   └── utils.py       # Preprocessing and metrics
├── requirements.txt   # Python dependencies
└── README.md          # Project overview
```

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Train
```bash
python main.py \
  --train_imgs data/train/images \
  --train_masks data/train/masks \
  --val_imgs data/val/images \
  --val_masks data/val/masks \
  --epochs 50
```

---

## 📈 Evaluation Metrics
- **Dice Similarity Coefficient (DSC)**
- **Jaccard Index (IoU)**

Evaluated against manually segmented ground truth annotations.

---

## 🗃️ Data Format
- Input format: NIfTI (`.nii`)
- Shape: `(96, 512, 512)` per volume
- Preprocessing: skull-stripping, normalization, resizing

---