# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-99.9%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Project 1: Artificial Neural Network (ANN) Classification

- **Description:**  
  Developed two ANN classification models addressing different prediction tasks:

  1. **Income Level Prediction (TensorFlow/Keras):**  
     - Dataset features include normalized numerical values like age, fnlwgt, education_num, capital_gain/loss, hours_per_week, and label (>50K or ≤50K).  
     - Model architecture: Sequential with 3 Dense layers (6, 8, 1 units) and Dropout layers to reduce overfitting.  
     - Model size: 107 parameters.  
     - Performance:  
       - Cross-validation scores ranged from 45% to 61%.  
       - Final test accuracy improved to ~72%, with test loss around 0.52.

  2. **Credit Eligibility Prediction (PyTorch):**  
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
  - TensorFlow ANN model: [`ANN_model.py`](ANN_model.py), [`ANN_usage.py`](ANN_usage.py)  
  - PyTorch ANN model: [`ANN_model2.py`](ANN_model2.py)  
  - Deployment configuration: [`Dockerfile`](Dockerfile)  
  - Detailed final report: [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing)  
    *(Report includes step-by-step process from data preprocessing to model evaluation and comparison, integrating RapidMiner and Python)*

- **Summary:**  
The two ANN models successfully classify their respective datasets with good accuracy, demonstrating effective feature preprocessing, model design, and training strategies. The PyTorch model, in particular, shows robust training stability with ~88% accuracy, while the TensorFlow model improves test accuracy from initial low cross-validation scores to 72%. Both models provide a strong baseline for ANN classification tasks with tabular data.


## Project 2: Convolutional Neural Network (CNN) for Handwritten Digit Recognition

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
  - Comprehensive final report: [`ANN and CNN Report (PDF)`](https://drive.google.com/file/d/1T1dkZxAcpdSUJ2gxWtfwASa8cqKCNaHt/view?usp=sharing)  
    *(Report includes detailed model design, training pipeline, performance evaluation, and comparison with baseline methods using RapidMiner and Python)*

- **Summary:**
  The CNN model effectively classifies handwritten digits with excellent accuracy, benefiting from convolutional feature extraction and dropout regularization. The model's high
  accuracy (>98%) on test data confirms its robustness and suitability for image classification tasks in real-world scenarios.

## Project 3: Image Classification with Pretrained ResNet-50d 🖼️

- **Description**:  
  This prototype demonstrates the implementation of an image classification system using the **ResNet-50d** pretrained model. It applies transfer learning on an ImageNet-style dataset to classify visual objects efficiently. The goal is to evaluate the feasibility and accuracy of using high-performance pretrained CNNs in custom classification scenarios.

- **The prototype focuses on**:
  - Loading and preprocessing input images
  - Using `timm` to load and fine-tune ResNet-50d
  - Performing inference and evaluating results

- **Libraries Used**:
  - **Data Analysis:** `pandas`, `NumPy`
  - **Visualization:** `matplotlib`
  - **Image Handling:** `PIL`
  - **Pretrained Model & Training:** `pytorch`, `torchvision`, `timm`

- **Deliverables**:
  - Hands-on Demo Notebook: [`DL_1_Classified_object_imagenet.ipynb`](DL_1_Classified_object_imagenet.ipynb)


## Project 4: Object Detection Using YOLOv8n 🚦

- **Description**:  
  This prototype implements an object detection pipeline using **YOLOv8n**, a lightweight and real-time object detection model from Ultralytics. The system is designed to detect and localize multiple object classes within static images, offering practical insights into integrating YOLOv8 models in detection-based workflows.

- **The prototype scope includes**:
  - Loading pretrained YOLOv8n weights
  - Running inference on test images
  - Visualizing detections with bounding boxes and confidence scores

- **Libraries Used**:
  - **Data Analysis:** `pandas`, `NumPy`
  - **Image Handling:** `PIL`
  - **Pretrained Model:** `YOLOv8n`
  - **Deep Learning & Model Inference:** `pytorch`, `ultralytics`, `yolo`

- **Deliverables**:
  - Hands-on Demo Notebook: [`DL_2_Oblect_dectection.ipynb`](DL_2_Oblect_dectection.ipynb)


## Project 5: Fine-Tuning YOLOv8n for Smoke Detection 🔥

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


## Project 6: Image Search from Video using CLIP 🎥🔍

- **Description**:  
  A prototype project showcasing **image-based and text-based search** across frames extracted from a YouTube video using the **CLIP (Contrastive Language–Image Pre-training)** model. The system supports relevance feedback to improve search results using label propagation.

- **Key Features**:
  - Download and extract frames from YouTube videos using `yt-dlp`
  - Use OpenAI’s CLIP model to embed both text and images into a shared space
  - Perform image search using:
    - Text queries (e.g., "a red car")
    - Image queries (e.g., search using a reference image)
  - Interactive relevance feedback:
    - Mark search results as relevant/irrelevant
    - Use **label propagation** to update scores and refine results

- **Prototype Scope**:
  - Not a full application — serves as a **quick experimental demo** and proof-of-concept
  - Great starting point for a video content indexing system or media search engine

- **Libraries Used**:
  - **Data Handling**: `pandas`, `NumPy`
  - **Video & Image Handling**: `yt_dlp`, `cv2`, `PIL`
  - **Visualization**: `plotly`
  - **Deep Learning & Models**: `pytorch`, `torchvision`, `transformers`, `CLIP`
  - 
- **Deliverables**:
  - Demo Notebook: [`DL_4_Image_search.ipynb`](DL_4_Image_search.ipynb)

## Project 7: Thai LLM Model Usage with OpenThaiGPT 🇹🇭🧠

- **Description**:  
  A prototype project demonstrating how to use the **OpenThaiGPT-1.0.0-7B-Chat** model for Thai language understanding and generation. It includes a **prompt-based interface** implemented with **Gradio**, allowing users to interact with the Thai LLM via a simple web UI.

- **Key Features**:
  - Loads and runs **OpenThaiGPT-7B-chat** (Hugging Face model) using local inference tools
  - Supports **prompt-based interaction** with Thai language queries
  - Simple and responsive **Gradio UI** for interactive chat interface
  - Lays the foundation for Thai-language applications like virtual assistants, summarizers, or translators

- **Prototype Scope**:
  - Early-stage prototype focusing on functionality rather than optimization
  - Intended as a learning and experimentation platform for Thai LLM integration

- **Libraries Used**:
  - **Data Handling**: `pandas`, `NumPy`
  - **Web Interface**: `gradio`
  - **LLM & Backend**:
    - `pytorch`
    - `llama-index` (for managing documents/chunks, if needed)
    - `llama-cpp-python` (for lightweight LLM execution)

- **Deliverables**:
  - Demo Notebook: [`DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb`](DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb)

## Project 8: Generative and Summarization Tasks Using LLMs ✍️📄

- **Description**:  
  This project demonstrates a simple prototype for **text generation** and **summarization** using popular pretrained LLMs:
  - **Text Generation**: Utilizes the `GPT-2` model for generating free-form text based on custom prompts.
  - **Text Summarization**: Uses `facebook/bart-large-cnn` to summarize long text inputs into concise summaries.

- **Key Features**:
  - Implements two core NLP tasks using pretrained models via the `transformers` library
  - Includes example functions for:
    - `generate_text(prompt)` – Generate creative or context-aware completions
    - `summarize_text(input_text)` – Extract key information in a shorter form
  - Supports experimentation with different decoding methods (e.g., top-k, top-p sampling)

- **Prototype Scope**:
  - Standalone notebook demonstrating usage
  - Ideal for content generation, article summarization, or chatbot backend

- **Libraries Used**:
  - **Model Access & Execution**: `pytorch`, `transformers`
  - **Models**:
    - `gpt2` (text generation)
    - `facebook/bart-large-cnn` (summarization)

- **Deliverables**:
  - Demo Notebook: [`DL_6_summarize_gen_text.ipynb`](DL_6_summarize_gen_text.ipynb)

## Project 9: OCR Extract Text from Picture 📝

- **Description**:  
  This project uses Tesseract OCR to extract text from images sourced via URLs. The solution includes a Flask web application deployed with Docker on AWS EC2. Continuous Integration and Deployment (CI/CD) is managed through GitHub Actions, ensuring streamlined updates.

- **Libraries Used**:
  - Image Processing: `opencv-python`
  - Optical Character Recognition (OCR): `pytesseract`
  - Web Development & Networking: `Flask`, `Gunicorn`, `Nginx`
  - Containers & Deployment: `Docker`, `docker-compose`
  - CI/CD: GitHub Actions
  - Deployment Platform: AWS EC2

- **Resources**:
  - Repository: [GitHub - OCR1_project](https://github.com/JPP-J/OCR1_project.git)
  - Code Demo Prototype Notebook: [Notebook](DL_7_OCR.ipynb)


## Project 10: Object Detection Realtime

- **Description**:  
  Capture screen recording video from a personal CCTV setup via an application and integrate with YOLOv8n to perform real-time object detection.

- **Libraries Used**:
  - Data Analysis: `pandas`, `NumPy`
  - Image Processing: `opencv-python` (`cv2`)
  - Pretrained Model: YOLOv8n
  - Deep Learning Frameworks: `pytorch`, `ultralytics`, `yolo`

- **Resources**:  
  - Repository: [GitHub - object_detection_realtime_project](https://github.com/JPP-J/object_dectection_realtime_project.git)

## Project 11: GEMINI-API Project
- **Description**: Hands-on Python demo showcasing a chat application using the GEMINI-API AI. Supports preserving conversation history to maintain context across chat sessions.
- **Libraries Used**:
  - Data Handling: `json`
  - Environment Configuration: `dotenv`
  - AI & Deep Learning: `google-genai`
- **Model Used**: `gemini-2.0-flash`
- **Features**:
  - Manages API keys securely via environment variables
  - Preserves past chat history for context-aware AI responses
- **Resources**:
  - Repository: [GitHub - DL-2_GEMINI_project](https://github.com/JPP-J/DL-2_GEMINI_project)


## Project 12: Chat with LLM Projects 💬

- **Description**:  
  Hands-on Python prototype showcasing an interactive chat application using the Qwen and Qwen2.5 LLMs. Built with GPU acceleration (`torch-cuda`) and powered by `transformers`, it includes reusable example functions to demonstrate message handling, response generation, and history maintenance.

- **Prototype Scope**:  
  This prototype demonstrates how to build an interactive chat system using large language models (LLMs) with preserved message history. It simulates human-like conversations by maintaining context between turns and generating context-aware replies.

- **Libraries Used**:
  - **Processing:** `torch-cuda`
  - **Deep Learning/AI:** `transformers`

- **Models**: Qwen, Qwen2.5

- **Features**:
  - Preserves multi-turn chat history for continuity
  - Example functions to send/receive messages
  - Supports conversational AI experimentation

- **Deliverables**:
  - Hands-on Demo Notebook: [`DL_8_chat_LLM.ipynb`](DL_8_chat_LLM.ipynb)


## Project 13: Prompt-based Generation with LLM Projects ✨

- **Description**:  
  A detailed Python prototype showing multiple methods to perform text generation via prompting. Includes:
  - Loading models/tokenizers directly for custom control
  - Using `transformers.pipeline()` for quick abstraction
  - The notebook offers modular functions and examples for each generation approach.

- **Prototype Scope**:  
  This prototype explores how to generate coherent text using various prompt-based techniques with Qwen/Qwen2.5 LLMs. It demonstrates direct and abstracted model usage to support flexible experimentation in language generation tasks.

- **Libraries Used**:
  - **Processing:** `torch-cuda`
  - **Deep Learning/AI:** `transformers`

- **Models**: Qwen, Qwen2.5

- **Features**:
  - Demonstrates direct and high-level generation methods
  - Covers best practices in prompt design
  - Includes usage examples for each method

- **Deliverables**:
  - Hands-on Demo Notebook: [`DL_9_Prompt_based_Generation.ipynb`](DL_9_Prompt_based_Generation.ipynb)



## Project 14: RAG-based Chatbot with Ollama and FAISS
- **Description**: Demo to build a Retrieval-Augmented Generation (RAG) chatbot with vector search via FAISS and context-aware responses through Ollama. Features document upload, persistent chat history, and a Gradio UI.
- **Libraries Used**:
  - Processing & NLP: `FAISS`, `pythainlp`
  - Deep Learning/AI: `langchain`, `HuggingFaceEmbeddings`, `ollama`
  - UI: `gradio`
  - DevOps: `Docker`, `Docker Compose`, `GitHub Actions`
- **Models**:
  - Ollama: `gemma3:1b`
  - Embeddings: `all-MiniLM-L6-v2`
- **Resources**:
  - Repository: [GitHub - DL-3_RAG](https://github.com/JPP-J/DL-3_RAG)


