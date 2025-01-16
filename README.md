# Deep learning Projects

This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Project 1: CNN model and ANN model Project
- **Description**: associated on datatset MNIST identified hand written picture of number 0-9 with CNN model and other dataset for ANN 
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - Visualization: matplotlib, seaborn
  - Image Handling: PIL
  - Model Handling: pickle, joblib
  - Deep Learning: tensorflow, keras
  - Model Evaluation: cross_val_score, Loss, Accuracy
- **Provides**:
    - datatset [MNIST training ](https://github.com/JPP-J/deep-_learning_project/tree/1e06ac3f6590fd3618c403e7f454b44b0299ea12/data/MNIST%20-%20JPG%20-%20training), [MNIST testing](https://github.com/JPP-J/deep-_learning_project/tree/1e06ac3f6590fd3618c403e7f454b44b0299ea12/data/MNIST%20-%20JPG%20-%20training) and other
  - example python code for deep learning data on tensorflow with:
      - ANN [ANN_model.py](https://github.com/JPP-J/deep-_learning_project/blob/1e06ac3f6590fd3618c403e7f454b44b0299ea12/ANN_model.py) and [ANN_usage.py](https://github.com/JPP-J/deep-_learning_project/blob/1e06ac3f6590fd3618c403e7f454b44b0299ea12/ANN_usage.py)
      - CNN [CNN_model.py](https://github.com/JPP-J/deep-_learning_project/blob/1e06ac3f6590fd3618c403e7f454b44b0299ea12/CNN_model.py) and [CNN_usage.py](https://github.com/JPP-J/deep-_learning_project/blob/1e06ac3f6590fd3618c403e7f454b44b0299ea12/CNN_usage.py)
  - rapid miner processes for deep learning data with:
      - ANN
      - CNN
  - [Deep Learning Project Report](https://drive.google.com/file/d/1z4KdqlYg3F4nBzVYaPEvUQAG2rQdejAC/view?usp=sharing) 

     
## Project 2: Image Classification Project
- **Description**: For image classification tasks on resnet50d model
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - Visualization: matplotlib
  - Image Handling: PIL
  - Pretrained Model: ResNet-50d
  - Deep Learning: pytorch, torchvision, timm
- [ImageNet Usage Project Notebook](https://github.com/JPP-J/deep-_learning_project/blob/c310f4a0ebcec18f773cb4cb3b62b42cc7c232ba/DL_1_Classified_object_imagenet.ipynb)

## Project 3: Object Detection Project
- **Description**:  Object Detection on yolov8n model
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - Image Handling: PIL
  - Pretrained Model:  YOLOv8n
  - Deep Learning: pytorch, ultralytics, yolo
- [Object detecion picture with YOLO Notebook](https://github.com/JPP-J/deep-_learning_project/blob/5dafda7bea3fadb6fafba5723149e65eac65f9e0/DL_2_Oblect_dectection.ipynb)

## Project 4: Fine-tuning pretrained model Project
- **Description**: Fine-tuning pretrained model with smoke dataset picture from roboflow and test accuracy achieved at 90%
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - Image Handling: PIL
  - Pretrained Model:  YOLOv8n
  - Deep Learning: pytorch, ultralytics, yolo
  - Computer Vision Tools: roboflow
  - Model Evaluation: Instances, Box(P), Box(R), mAP50(mean Average Precision at IoU = 0.5) and mAP50-95(mean Average Precision from IoU 0.5 to 0.95)
- [Train on custom dataset Notebook](https://github.com/JPP-J/deep-_learning_project/blob/5dafda7bea3fadb6fafba5723149e65eac65f9e0/DL_3_train_smoke_dataset.ipynb)

## Project 5: Image Search Project
- **Description**:  Image Search from video with CLIP Pre-trained Model
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - Download Video: yt_dlp
  - Image Handling: PIL, cv2
  - Pretrained Model:  CLIP Pre-trained
  - Deep Learning: pytorch, CLIP, torchvision, transformers
- [Image Search Notebook](https://github.com/JPP-J/deep-_learning_project/blob/5ae7f5701be9a6fe7e4e35cfa914196da49e2e93/DL_4_Image_search.ipynb)

## Project 6: Thai LLM model usage Project
- **Description**: Usage That LLM model with openthaigpt-1.0.0-7b-chat with hugging face
- **Libraries Used**:
  - Data Analysis: pandas, NumPy
  - GUI Interface: gradio
  - Pretrained Model:  openthaigpt-1.0.0-7b-chat
  - Deep Learning: pytorch, llama-index, llama-cpp-python
- [Promt with LLM openthaigpt_1_0_0_7b_chat model Notebook](https://github.com/JPP-J/deep-_learning_project/blob/74c992978381f462a2f8bed2aaf6009c5f58e732/DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb) 
