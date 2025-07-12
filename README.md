#  Image Classification with Pretrained ResNet-50d Project 🖼️
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project explores the use of **transfer learning** to implement an image classification system using the **ResNet-50d** architecture pretrained on **ImageNet**. It aims to demonstrate how high-performing pretrained models can be quickly adapted to new image classification tasks with minimal effort and strong performance.

### 🧩 Problem Statement

Training large convolutional neural networks from scratch is time-consuming and requires vast labeled datasets. For many use cases—especially where data is limited—transfer learning offers a practical alternative. This project investigates the feasibility and accuracy of adapting ResNet-50d to a new, custom image classification task.

### 🔍 Approach

By leveraging the **`timm`** library, the ResNet-50d model is loaded with pretrained weights. The final layer is fine-tuned to match the number of classes in the custom dataset. The pipeline includes preprocessing, training adjustments, and inference testing to evaluate model adaptability and performance.

### 🎢 Processes

1. **Data Loading** – Images are loaded from an ImageNet-style directory structure  
2. **Preprocessing** – Resizing, normalization, and batch formatting for input compatibility  
3. **Model Setup** – Load ResNet-50d with pretrained weights using `timm`  
4. **Fine-Tuning** – Adjust and retrain the classifier head for custom labels  
5. **Evaluation** – Predict on sample images, analyze classification results, visualize outputs  
6. **Deployment Demo** – Provided Jupyter Notebook for interactive testing

### 🎯 Results & Impact

- Achieved **high prediction accuracy** on test images from the custom dataset  
- Showcased strong **generalization** capabilities with minimal retraining  
- Proved the practicality of using **ResNet-50d** for real-world image classification tasks through transfer learning

### ⚙️ Model Development Challenges

- **Input Format Compatibility** – Ensured correct input dimensions and normalization for pretrained model expectations  
- **Avoiding Overfitting** – Used early stopping, controlled learning rate, and considered freezing lower layers  
- **Hardware Optimization** – Optimized batch size and utilized GPU acceleration for smoother training and inference

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

---
