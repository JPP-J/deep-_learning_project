# Fine-Tuning YOLOv8n for Smoke Detection Project 🔥
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project focuses on fine-tuning the lightweight and high-performance **YOLOv8n** object detection model for real-time **smoke detection** using a custom dataset from **Roboflow**. The goal is to enable early identification of smoke in images, which is critical for industrial safety, wildfire monitoring, and fire prevention systems.

### 🧩 Problem Statement

Detecting smoke early can significantly reduce the risk of fire-related disasters. Manual monitoring is error-prone and inefficient, especially in remote or large-scale industrial environments. This project addresses the need for a fast, accurate, and lightweight smoke detection system that can be deployed on edge devices or surveillance infrastructure.

### 🔍 Approach

Using the **Ultralytics YOLOv8** framework, the pretrained `yolov8n.pt` model was fine-tuned on a labeled smoke dataset. The workflow includes data augmentation, transfer learning, model evaluation, and performance validation.

### 🎢 Processes

1. **Dataset Preparation** – Imported smoke images from Roboflow with bounding box annotations  
2. **Training Configuration** – Customized `data.yaml` and hyperparameters using YOLOv8 CLI  
3. **Fine-Tuning** – Leveraged pretrained weights and trained using Tesla T4 GPU for acceleration  
4. **Evaluation** – Assessed performance using precision, recall, mAP metrics, and confusion matrix  
5. **Inference Demo** – Tested real-world images for visual validation of detection results

### 🎯 Results & Impact

- **Precision (Box(P))**: 0.986  
- **Recall (Box(R))**: 0.910  
- **mAP@0.5**: 0.965  
- **mAP@0.5:0.95**: 0.730  
- **Inference Speed**: ~5 ms/image (Tesla T4)  
- **Confusion Matrix**:
  
  
    |               | Pred: Smoke (0) | Pred: Background (1) |
    |---------------|------------------|-----------------------|
    | **True: Smoke (0)**     | 150                | 9                     |
    | **True: Background (1)**| 7                  | 0                     |
  
The model demonstrates high precision and fast inference, making it suitable for real-time smoke detection systems. Slight over-sensitivity to background is acceptable in safety-first contexts where false negatives (missed smoke) are more critical than false positives.

### ⚙️ Model Development Challenges

- **False Positives in Background** – Required careful tuning of confidence thresholds to reduce misclassification  
- **Bounding Box Accuracy** – Fine-tuned IoU thresholds to balance detection sensitivity vs. precision  
- **Data Quality & Imbalance** – Addressed class imbalance and ensured clean annotation quality  
- **Deployment Readiness** – Focused on lightweight YOLOv8n version to enable edge deployment without sacrificing detection speed


## **Libraries Used**:
  - **Data Analysis:** `pandas`, `NumPy`
  - **Image Handling:** `PIL`
  - **Deep Learning & Training:** `pytorch`, `ultralytics`, `yolo`
  - **Computer Vision Platform:** `Roboflow`
  - **Evaluation Metrics:** `Box(P)`, `Box(R)`, `mAP@0.5`, `mAP@0.5:0.95`, Confusion Matrix

## **Deliverables**:
  - Demo Notebook: [`DL_3_train_smoke_dataset.ipynb`](DL_3_train_smoke_dataset.ipynb)
  - Trained Model: `runs/detect/train2/weights/best.pt` in notebook 
  - Training Logs: `runs/detect/train2/` in notebook

----
