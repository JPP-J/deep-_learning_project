# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Image Search from Video using CLIP Project 🎥🔍

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
