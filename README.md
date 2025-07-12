# Image Search from Video using CLIP Project 🎥🔍
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project builds a prototype system for **image and text-based search** across frames extracted from YouTube videos. It uses **CLIP (Contrastive Language–Image Pre-training)** to embed images and text into a shared semantic space, enabling accurate similarity-based retrieval. Ideal for prototyping **video indexing**, **content tagging**, and **media search engines**.

### 🧩 Problem Statement

Finding specific visual content in long videos is time-consuming and difficult without proper indexing. This project addresses this problem by enabling **natural language or image queries** to locate relevant frames, dramatically improving search efficiency.

### 🔍 Approach

- Frames are extracted from videos using `yt-dlp` and processed through OpenAI’s **CLIP model**.
- Both **text** and **image queries** are embedded into the same vector space for comparison.
- Users can interactively refine results through **relevance feedback**, leveraging **label propagation** to improve ranking of relevant frames.

### 🎢 Processes

1. **Video Download & Frame Extraction** – Automatically download YouTube videos and extract frames at a fixed rate.
2. **CLIP Embedding** – Convert both image frames and search inputs (text/image) into vector embeddings using CLIP.
3. **Search** – Perform nearest-neighbor search in the embedding space.
4. **Relevance Feedback** – Use user-provided relevance annotations to adjust similarity scores via label propagation.
5. **Visualization** – Show ranked results and feedback updates using interactive plots.

### 🎯 Results & Impact

- The model retrieves semantically relevant video frames based on simple user inputs like `"a dog running"` or a reference image.
- Label propagation allows dynamic refinement of search results.
- Demonstrates the power of **zero-shot learning** using pretrained vision-language models in real-world multimedia applications.

### ⚙️ Model Development Challenges

- **Video Frame Quality** – Variability in frame clarity and content density affects search quality.
- **Embedding Clustering** – Semantically similar frames often cluster well, but dissimilar frames can sometimes appear near decision boundaries.
- **Relevance Feedback Integration** – Implemented a lightweight label propagation method to balance speed and precision without retraining CLIP.

## **Key Features**:
  - Download and extract frames from YouTube videos using `yt-dlp`
  - Use OpenAI’s CLIP model to embed both text and images into a shared space
  - Perform image search using:
    - Text queries (e.g., "a red car")
    - Image queries (e.g., search using a reference image)
  - Interactive relevance feedback:
    - Mark search results as relevant/irrelevant
    - Use **label propagation** to update scores and refine results

## **Prototype Scope**:
  - Not a full application — serves as a **quick experimental demo** and proof-of-concept
  - Great starting point for a video content indexing system or media search engine

## **Libraries Used**:
  - **Data Handling**: `pandas`, `NumPy`
  - **Video & Image Handling**: `yt_dlp`, `cv2`, `PIL`
  - **Visualization**: `plotly`
  - **Deep Learning & Models**: `pytorch`, `torchvision`, `transformers`, `CLIP`

## **Deliverables**:
  - Demo Notebook: [`DL_4_Image_search.ipynb`](DL_4_Image_search.ipynb)

---
