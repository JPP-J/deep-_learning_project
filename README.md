# Convolutional Neural Network (CNN) for Handwritten Digit Recognition Project 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project focuses on building a **Convolutional Neural Network (CNN)** model to accurately classify handwritten digits (0–9) using the **MNIST dataset**. It leverages deep learning techniques to extract spatial hierarchies in image data, making it well-suited for visual pattern recognition tasks.

### 🧩 Problem Statement

Handwritten digit recognition plays a key role in various real-world applications like postal code reading, form processing, and bank check validation. Manual recognition is inefficient and error-prone—this project aims to automate the process using deep learning to achieve high accuracy and reliability.

### 🔍 Approach

The model was developed using **TensorFlow** and **Keras**, implementing a lightweight yet effective CNN architecture. Convolutional and pooling layers enable hierarchical feature extraction, while dropout regularization reduces overfitting and improves generalization.

### 🎢 Processes

1. **Data Loading & Preprocessing** – Normalized pixel values, reshaped image tensors, and encoded labels  
2. **Model Design** – Built a 3-layer CNN with increasing filter sizes and ReLU activation, followed by dense and dropout layers  
3. **Model Training** – Trained the model over multiple epochs using the Adam optimizer and categorical cross-entropy loss  
4. **Performance Evaluation** – Measured training/validation accuracy and loss curves, and assessed final model on test set  
5. **Deployment Ready** – Model saved for reuse, Docker setup included for containerized deployment  
6. **Reporting** – Final deliverables include a detailed report, codebase, and visualized results

### 🎯 Results & Impact

- **Training Accuracy:** 99.32%  
- **Validation Accuracy:** 98.81%  
- **Test Accuracy:** 98.97%  
- **Model Size:** ~364.5 KB (93,322 trainable parameters)  

These results indicate the model is robust, efficient, and ready for deployment in small-scale image classification systems or embedded AI applications where resource efficiency matters.

### ⚙️ Model Development Challenges

Key challenges included:

- **Preventing Overfitting:** Addressed using dropout and monitoring validation loss  
- **Model Simplicity vs. Accuracy:** Designed a compact architecture that still achieves near state-of-the-art performance  
- **Hyperparameter Tuning:** Balanced filter counts, dropout rates, and learning rate for optimal performance  
- **Deployment Considerations:** Optimized model size and dependencies for Docker containerization

## Model Architecture:
  - Conv2D (32 filters, 3x3 kernel) → MaxPooling2D (2x2)  
  - Conv2D (64 filters, 3x3 kernel) → MaxPooling2D (2x2)  
  - Conv2D (64 filters, 3x3 kernel)  
  - Flatten layer to convert 3D feature maps to 1D vector  
  - Dense layer (64 units) with dropout (for overfitting prevention)  
  - Output Dense layer (10 units) with softmax activation for multi-class classification


## **Model Parameters:**  
  - Total parameters: 93,322 (~364.5 KB)  
  - All parameters are trainable.

## **Libraries and Tools Used:**  
  - Data processing and visualization: `pandas`, `NumPy`, `matplotlib`, `seaborn`  
  - Image processing: `PIL`  
  - Deep learning: `tensorflow`, `keras`  
  - Model evaluation and persistence: `pickle`, `joblib`

## **Deliverables:**  
  - MNIST dataset folders for training and testing: [`MNIST Training`](data/MNIST%20-%20JPG%20-%20training), [`MNIST Testing`](data/MNIST%20-%20JPG%20-%20testing)  
  - CNN model implementation: [`CNN_model.py`](CNN_model.py), [`CNN_usage.py`](CNN_usage.py)  
  - Deployment setup: [`Dockerfile`](Dockerfile)
  - Example Result: [`example_results.txt`](example_results.txt)
  - Comprehensive final report: [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing)  
    *(Report includes detailed model design, training pipeline, performance evaluation, and comparison with baseline methods using RapidMiner and Python)*

## **Summary:**
  The CNN model effectively classifies handwritten digits with excellent accuracy, benefiting from convolutional feature extraction and dropout regularization. The model's high
  accuracy (>98%) on test data confirms its robustness and suitability for image classification tasks in real-world scenarios.

---
