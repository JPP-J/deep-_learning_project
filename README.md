# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Convolutional Neural Network (CNN) for Handwritten Digit Recognition Project

- **Description:**  
  Developed a CNN model using **TensorFlow** and **Keras** to classify handwritten digits (0–9) from the **MNIST dataset**.  
  The architecture consists of 3 convolutional layers with ReLU activation and max-pooling, followed by a fully connected dense layer and dropout for regularization.  
  The model achieved near state-of-the-art accuracy with strong generalization on test data.

- **Model Architecture:**  
  - Conv2D (32 filters, 3x3 kernel) → MaxPooling2D (2x2)  
  - Conv2D (64 filters, 3x3 kernel) → MaxPooling2D (2x2)  
  - Conv2D (64 filters, 3x3 kernel)  
  - Flatten layer to convert 3D feature maps to 1D vector  
  - Dense layer (64 units) with dropout (for overfitting prevention)  
  - Output Dense layer (10 units) with softmax activation for multi-class classification

- **Model Parameters:**  
  - Total parameters: 93,322 (~364.5 KB)  
  - All parameters are trainable.

- **Performance Metrics:**  
  - Final training accuracy: **99.32%**  
  - Final validation accuracy: **98.81%**  
  - Training loss: 0.0223  
  - Validation loss: 0.0542  
  - Test set accuracy: **98.97%**

- **Libraries and Tools Used:**  
  - Data processing and visualization: `pandas`, `NumPy`, `matplotlib`, `seaborn`  
  - Image processing: `PIL`  
  - Deep learning: `tensorflow`, `keras`  
  - Model evaluation and persistence: `pickle`, `joblib`

- **Deliverables:**  
  - MNIST dataset folders for training and testing: [`MNIST Training`](data/MNIST%20-%20JPG%20-%20training), [`MNIST Testing`](data/MNIST%20-%20JPG%20-%20testing)  
  - CNN model implementation: [`CNN_model.py`](CNN_model.py), [`CNN_usage.py`](CNN_usage.py)  
  - Deployment setup: [`Dockerfile`](Dockerfile)
  - Example Result: [`example_results.txt`](example_results.txt)
  - Comprehensive final report: [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing)  
    *(Report includes detailed model design, training pipeline, performance evaluation, and comparison with baseline methods using RapidMiner and Python)*

- **Summary:**
  The CNN model effectively classifies handwritten digits with excellent accuracy, benefiting from convolutional feature extraction and dropout regularization. The model's high
  accuracy (>98%) on test data confirms its robustness and suitability for image classification tasks in real-world scenarios.
