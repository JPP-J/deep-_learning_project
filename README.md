# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Fine-Tuning YOLOv8n for Smoke Detection Project 🔥

- **Description**:  
  Fine-tuned a pretrained **YOLOv8n** model on a custom **smoke detection dataset** from **Roboflow**, aiming to detect visible smoke in real-world images. The model was trained using the Ultralytics YOLOv8 framework and validated on a GPU (Tesla T4), delivering robust results suitable for industrial safety and early fire detection systems.

- **Results**:  
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
  
    - The model performs very well in identifying smoke (150/159 correct), but tends to **misclassify background as smoke**, indicating high sensitivity — appropriate for safety-critical use cases where missing smoke is riskier than a false positive.

- **Libraries Used**:
  - **Data Analysis:** `pandas`, `NumPy`
  - **Image Handling:** `PIL`
  - **Deep Learning & Training:** `pytorch`, `ultralytics`, `yolo`
  - **Computer Vision Platform:** `Roboflow`
  - **Evaluation Metrics:** `Box(P)`, `Box(R)`, `mAP@0.5`, `mAP@0.5:0.95`, Confusion Matrix

- **Deliverables**:
  - Demo Notebook: [`DL_3_train_smoke_dataset.ipynb`](DL_3_train_smoke_dataset.ipynb)
  - Trained Model: `runs/detect/train2/weights/best.pt` in notebook 
  - Training Logs: `runs/detect/train2/` in notebook
