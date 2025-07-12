# Artificial Neural Network (ANN) Classification Project 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This deep learning project develops and evaluates two **Artificial Neural Network (ANN)** models using different frameworks—**TensorFlow (Keras)** and **PyTorch**—to solve two real-world classification problems: income level prediction and credit eligibility assessment. Both problems have direct applications in financial customer segmentation for loan credit evaluation.

### 🧠 TensorFlow ANN – Income Level Prediction
---

#### 🧩 Problem Statement

To predict whether an individual’s income is **greater than \$50K** or **less than or equal to \$50K**, based on demographic and financial data. This prediction can help financial institutions assess customer income segments for loan eligibility.

#### 🔍 Approach

- Built with **TensorFlow/Keras Sequential API**
- Used **normalized numerical features**: `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
- Target label: `>50K` or `≤50K`
- Model architecture:
  - Input → Dense(64) → Dropout → Dense(64) → Dropout → Dense(64) → Dropout → Dense(1) (sigmoid)
  - Total Parameters: **38,981**
- Loss function: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy

#### 🎯 Results

- Cross-validation accuracy: **70% – 77%**  
- Final test accuracy: **~70%**  
- Model generalized better after dropout regularization  
- Demonstrated a lightweight ANN architecture suitable for small-to-medium scale income classification tasks  

This model can assist team strategy by predicting the income bracket of each customer, which supports targeted loan offer segmentation. For example, financial institutions can use the predictions to determine appropriate loan levels or customize product offerings based on predicted income categories.



### 🔥 PyTorch ANN – Credit Eligibility Prediction
---

#### 🧩 Problem Statement

To classify whether a client is **credit eligible** based on various personal and financial attributes. This is particularly useful in customer segmentation for banking and credit scoring.

#### 🔍 Approach

- Developed with **PyTorch**
- Dataset includes **categorical** and **numerical** features (e.g., `job`, `marital`, `education`, `balance`, `housing`, `loan`)
- Target label: `yes` or `no`
- Preprocessing:
  - Missing value imputation (e.g., `age`)
  - Label encoding and scaling
- Model architecture:
  - Fully connected feedforward network:
    - **Input (11) → Hidden (6) → Hidden (6) → Output (1)**
  - Dropout layers to reduce overfitting
  - Mixed precision training enabled for speed
- Loss function: Binary Cross Entropy  
- Optimizer: Adam  
- Batch size: Moderate (not specified)  
- Epochs: Configurable via script

#### 🎯 Results

- Achieved stable testing accuracy of approximately **88%**
- Generated reliable and consistent binary predictions
- Trained efficiently with **mixed precision**, ensuring optimal performance and resource usage
- Proven effective for real-world **credit eligibility classification** tasks in financial applications

### ⚙️ Key Challenges

- **Overfitting** on limited features: Mitigated with dropout and regularization
- **Imbalanced datasets**: Required careful performance evaluation beyond accuracy
- **Cross-framework learning**: Showcases strengths of TensorFlow vs. PyTorch
- **Model comparison**: Evaluating lightweight vs. robust ANN architectures


## 🧰 Tools & Libraries Used

- **Data processing**: `pandas`, `NumPy`, `matplotlib`, `seaborn`
- **Deep learning**:
  - `TensorFlow`, `Keras` for income prediction
  - `PyTorch` for credit eligibility classification
- **Model evaluation**: `cross_val_score`, confusion matrix, accuracy
- **Persistence & Deployment**: `pickle`, `joblib`, `Dockerfile`


## 📎 Deliverables

| Component                        | Description                                                 |
|----------------------------------|-------------------------------------------------------------|
| `ANN_model_tf.py`               | TensorFlow ANN model definition                            |
| `ANN_usage_tf.py`               | TensorFlow inference script                                |
| `ANN_model_pt.py`               | PyTorch ANN model implementation                           |
| `Dockerfile`                    | Deployment setup                                           |
| `example_results.txt`           | Sample model outputs                                       |
| [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing) | Full model explanation and comparison (RapidMiner + Python) |



## Summary
The two ANN models successfully classify their respective datasets with good accuracy, demonstrating effective feature preprocessing, model design, and training strategies. The PyTorch model, in particular, shows robust training stability with ~88% accuracy, while the TensorFlow model improves test accuracy 70%. Both models provide a strong baseline for ANN classification tasks with tabular data.

*This project showcases practical deep learning applications using both **TensorFlow** and **PyTorch** in real-world classification problems, providing insights into model development, evaluation, and deployment pipelines.*

---
