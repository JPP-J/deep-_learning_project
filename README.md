# ## Image Classification with Pretrained ResNet-50d Project 🖼️
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project explores the application of **transfer learning** for image classification using the **ResNet-50d** model, a high-performance convolutional neural network architecture pretrained on **ImageNet**. The aim is to classify images efficiently and evaluate the viability of leveraging pretrained CNNs for custom datasets.

### 🧩 Problem Statement

Training deep convolutional networks from scratch requires vast data and computational resources. For many real-world scenarios—especially with limited labeled data—fine-tuning a pretrained model offers a practical solution. This project demonstrates how to use a pretrained ResNet-50d model for accurate object classification on custom image data.

### 🔍 Approach

The solution uses the **`timm`** library to load the ResNet-50d architecture with pretrained ImageNet weights. Transfer learning is applied by modifying and fine-tuning the model for the target dataset. The workflow includes image preprocessing, batch inference, and performance evaluation.

### 🎢 Processes

1. **Data Preparation** – Load images in ImageNet-style folder structure, resize and normalize inputs  
2. **Model Loading** – Load ResNet-50d from `timm` with pretrained weights  
3. **Fine-tuning** – Replace final classification layer, freeze base layers if necessary, and retrain on custom classes  
4. **Evaluation** – Use classification metrics and visual inspection of predicted labels  
5. **Inference Demo** – Perform single image predictions for hands-on verification  

### 🎯 Results & Impact

- **High accuracy** in top-1 predictions on custom dataset samples  
- **Transfer learning** reduced training time and improved generalization with minimal labeled data  
- Demonstrated the effectiveness of **ResNet-50d** in real-world classification pipelines

### ⚙️ Model Development Challenges

- **Data Format Compatibility:** Ensured dataset structure and input pipeline matched ImageNet conventions  
- **Overfitting Risk:** Controlled via early stopping and learning rate tuning during fine-tuning  
- **Hardware Efficiency:** Optimized batch size and used GPU acceleration for faster training and inference


## **The prototype focuses on**:
  - Loading and preprocessing input images
  - Using `timm` to load and fine-tune ResNet-50d
  - Performing inference and evaluating results

## **Libraries Used**:
  - **Data Analysis:** `pandas`, `NumPy`
  - **Visualization:** `matplotlib`
  - **Image Handling:** `PIL`
  - **Pretrained Model & Training:** `pytorch`, `torchvision`, `timm`

## **Deliverables**:
  - Hands-on Demo Notebook: [`DL_1_Classified_object_imagenet.ipynb`](DL_1_Classified_object_imagenet.ipynb)

*This prototype confirms the feasibility of integrating pretrained CNNs into lightweight image classification applications with minimal training overhead.*
---
