# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-99.9%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Project 1: ANN model Project 
- **Description**: Building ANN model associated on individual income dataset(to defines target label >50k or <=50k) for [ANN_model.py](ANN_model.py) with tensorflow(keras) and individual credit bank dataset (to defines target label yes or no for credit ability) [ANN_model2.py](ANN_model2.py) with pytorch can acheived accuracy both tarin and validation stage up to 88.00%
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Visualization: `matplotlib`, `seaborn`
  - Model Handling: `pickle`, `joblib`
  - Deep Learning: `tensorflow`, `keras`, `pytorch`
  - Model Evaluation: cross_val_score, Loss, Accuracy
- **Provides**:
  - [ANN_model.py](ANN_model.py) and [ANN_usage.py](ANN_usage.py) (tensorflow building model)
  - [ANN_model2.py](ANN_model2.py) (pytorch building model)
  - [Dockerfile](Dockerfile)
  - [ANN and CNN Report](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing) (rapid miner processes and above python code demo results)

## Project 2: CNN model Project
- **Description**: Building CNN model to classified image associated on datatset MNIST identified hand written picture of number 0-9 with tensorflow achieved accuracy of prediction up to 99.00%
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Visualization: `matplotlib`, `seaborn`
  - Image Handling: `PIL`
  - Model Handling: `pickle`, `joblib`
  - Deep Learning: `tensorflow`, `keras`
  - Model Evaluation: cross_val_score, Loss, Accuracy
- **Provides**:
  - datatset [MNIST training ](data/MNIST%20-%20JPG%20-%20training), [MNIST testing](data/MNIST%20-%20JPG%20-%20training) 
  -  CNN [CNN_model.py](CNN_usage.py) (tensorflow building model)
  - [Dockerfile](Dockerfile)
  - [ANN and CNN Report](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing) (rapid miner processes and above python code demo results)

     
## Project 3: Image Classification Project 🖼️
- **Description**: For image classification tasks on resnet50d model
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Visualization: `matplotlib`
  - Image Handling: `PIL`
  - Pretrained Model: ResNet-50d
  - Deep Learning: `pytorch`, `torchvision`, `timm`
- [Hand on code demo Notebook](DL_1_Classified_object_imagenet.ipynb)

## Project 4: Object Detection Project
- **Description**:  Object Detection on yolov8n model
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Image Handling: `PIL`
  - Pretrained Model:  YOLOv8n
  - Deep Learning: `pytorch`, `ultralytics, yolo`
- [Hand on code demo Notebook](DL_2_Oblect_dectection.ipynb)

## Project 5: Fine-tuning pretrained object dectection model Project
- **Description**: Fine-tuning pretrained model with smoke dataset picture from roboflow and test accuracy achieved at 90%
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Image Handling: `PIL`
  - Pretrained Model:  YOLOv8n
  - Deep Learning: `pytorch`, `ultralytics`, `yolo`
  - Computer Vision Tools: `roboflow`
  - Model Evaluation: Instances, Box(P), Box(R), mAP50(mean Average Precision at IoU = 0.5) and mAP50-95(mean Average Precision from IoU 0.5 to 0.95)
- [Hand on code demo Notebook](DL_3_train_smoke_dataset.ipynb)

## Project 6: Image Search Project
- **Description**:  Image Search from video with CLIP Pre-trained Model
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Download Video: `yt_dlp`
  - Image Handling: `PIL`, `cv2`
  - Pretrained Model:  CLIP Pre-trained
  - Deep Learning: `pytorch`, `CLIP`, `torchvision`, `transformers`
- [Hand on code demo Notebook](DL_4_Image_search.ipynb)

## Project 7: Thai LLM model usage Project
- **Description**: Usage That LLM model with openthaigpt-1.0.0-7b-chat with hugging face
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - GUI Interface: `gradio`
  - Pretrained Model:  openthaigpt-1.0.0-7b-chat
  - Deep Learning: `pytorch`, `llama-index`, `llama-cpp-python`
- [Hand on code demo Notebook](DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb)

## Project 8: Generative and Summarize tasks for text using model LLM Project
- **Description**: Usage That LLM model gpt2 for generative text task and facebook/bart-large-cnn model for summarize text task
- **Libraries Used**:
  - Pretrained Model: gpt2 and facebook/bart-large-cnn
  - Deep Learning: `pytorch`, `transformers`
- [Hand on code demo Notebook](DL_6_summarize_gen_text.ipynb)

## Project 9: OCR extract text from picture
- **Description**: Usage tessaract for extract text task from picture link
- **Libraries Used**:
  - image processing: `opencv-python`
  - Optical Character Recognition (OCR): `pytesseract`
  - web development and networking:  `flask`, `flask-ngrok`, `pyngrok`
  - WSGI (Web Server Gateway Interface): `Gunicorn`, `Nginx`
  - Deployment on: EC2-AWS
- [Repository](https://github.com/JPP-J/OCR1_project.git)
- [Hand on code demo Notebook](DL_7_OCR.ipynb)

## Project 10: Object Dectection Realtime
- **Description**: screen recording video from personal CCTV through application and integrate with yolov8n to do object detection in realtime 
- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - image processing: `opencv-python/cv2`
  - Pretrained Model:  yolov8n
  - Deep Learning: `pytorch`, `ultralytics`, `yolo`
- [Repository](https://github.com/JPP-J/object_dectection_realtime_project.git)

## Project 11: GEMINI-API Projects
- **Description**: hands on python code demo to create chat with GEMINI-API AI with preserve history chat of previously chat
  - Data Handling: `json`
  - Environment Handling: `dotenv`
  - Deep Learning/AI: `google-genai`
  - model: gemini-2.0-flash
- [Repository](https://github.com/JPP-J/DL-2_GEMINI_project)

## Project 12: Chat with LLM Projects
- **Description**: hands on python code demo to create chat with LLM model with preserve history chat of previously chat
  - Processing: `torch-cuda`
  - Deep Learning/AI: `transformers`
  - Model: Qwen/Qwen2.5
- [Hand on code demo Notebook](DL_8_chat_LLM.ipynb)

## Project 13: Prompt-based Generation with LLM Projects
- **Description**: hands on python code demo to Prompt-based Generation with Qwen/Qwen2.5 model
  - Processing: `torch-cuda`
  - Deep Learning/AI: `transformers`
  - model: Qwen/Qwen2.5
- [Hand on Code demo Notebook](DL_9_Prompt_based_Generation.ipynb)
