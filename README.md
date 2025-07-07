# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Artificial Neural Network (ANN) Classification Project

- **Description:**  
  Developed two ANN classification models addressing different prediction tasks:

  1. **Income Level Prediction (TensorFlow/Keras):**
     - **Problem**: To classified individual income Level, this can be one of attribute applied to segmentation of level of customer for loan credit.
     - Dataset features include normalized numerical values like age, fnlwgt, education_num, capital_gain/loss, hours_per_week, and label (>50K or ≤50K).  
     - Model architecture: Sequential with 3 Dense layers (6, 8, 1 units) and Dropout layers to reduce overfitting.  
     - Model size: 107 parameters.  
     - Performance:  
       - Cross-validation scores ranged from 45% to 61%.  
       - Final test accuracy improved to ~72%, with test loss around 0.52.
         
  3. **Credit Eligibility Prediction (PyTorch):**
     - **Problem**: To classified Credit Eligibility, this can be one of attribute applied to segmentation of level of customer for loan credit.
     - Dataset includes categorical and numerical features from bank marketing data (e.g., job, marital, education, balance).  
     - Missing age values handled; categorical and numerical preprocessing applied.  
     - Model architecture: Fully connected network with layers of sizes [11 → 6 → 6 → 1], with dropout for regularization.  
     - Training utilized mixed precision for efficiency.  
     - Training/Validation accuracy stabilized at ~88%.  
     - Final predictions indicate consistent binary outputs.

- **Libraries and Tools Used:**  
  - Data manipulation and visualization: `pandas`, `NumPy`, `matplotlib`, `seaborn`  
  - Model development: `tensorflow`, `keras`, `pytorch`  
  - Model evaluation and persistence: `cross_val_score`, `pickle`, `joblib`  
  - Mixed precision training to optimize performance (PyTorch model).

- **Deliverables:**  
  - TensorFlow ANN model: [`ANN_model_tf.py`](ANN_model_tf.py), [`ANN_usage_tf.py`](ANN_usage_tf.py)  
  - PyTorch ANN model: [`ANN_model_pt.py`](ANN_model_pt.py)  
  - Deployment configuration: [`Dockerfile`](Dockerfile)
  - Example Result: [`example_results.txt`](example_results.txt)
  - Detailed final report: [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing)  
    *(Report includes step-by-step process from data preprocessing to model evaluation and comparison, integrating RapidMiner and Python)*

- **Summary:**  
The two ANN models successfully classify their respective datasets with good accuracy, demonstrating effective feature preprocessing, model design, and training strategies. The PyTorch model, in particular, shows robust training stability with ~88% accuracy, while the TensorFlow model improves test accuracy from initial low cross-validation scores to 72%. Both models provide a strong baseline for ANN classification tasks with tabular data.

