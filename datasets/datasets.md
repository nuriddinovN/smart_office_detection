# 🗂️ Dataset Directory

This directory contains various images used **only for testing and evaluation purposes** in the **Smart Office Detection** project.

## 📌 Purpose

These samples are **not used for model training or fine-tuning**. Instead, they serve to:

- Evaluate the performance of object detection models.
- Visualize outputs during inference.
- Showcase real-world application cases for office environment understanding.

## ⚠️ Notes

- The model was **not fine-tuned** on this data.
- Instead of traditional fine-tuning, we used **prompt tuning** techniques on pre-trained models to adapt them for smart office scenarios.
- These images include screenshots, camera frames, and other test samples collected from publicly accessible sources or sample office footage.

## 📁 File Types

| Filename Example | Description                              |
|------------------|------------------------------------------|
| `*.jpg`          | Raw or extracted image samples           |
| `*.png`          | Screenshots of detection results         |
| `*_frame_*.jpg`  | Frames extracted from test videos        |

## 📷 Example Images

- `maxresdefault.jpg` – Sample frame from an online office video
- `photo_2025-08-01_10-18-51.jpg` – Mobile photo from a test session
- `Screenshot From 2025-08-02 19-26-28.png` – Inference result screenshot

---

Please do not use these images for training without proper data curation and labeling.
