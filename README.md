#  Thai LLM Model Usage with OpenThaiGPT 🇹🇭 Project 🧠

![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## 📌 Overview

This project demonstrates the use of **OpenThaiGPT-1.0.0-7B-Chat**, a Thai language large language model (LLM), for natural language understanding and generation. Through a simple **Gradio-based web interface**, users can interact with the model via prompt-based chat, enabling Thai-language conversational AI capabilities.

### 🧩 Problem Statement

Thai language NLP tools are less commonly available compared to English, limiting AI application development in the Thai language. This prototype aims to provide an accessible interface and integration example for Thai LLMs to foster research and practical deployments in areas like chatbots, summarization, and translation.

### 🔍 Approach

- Load and run the OpenThaiGPT-7B-chat model locally using PyTorch and lightweight LLM execution frameworks.
- Build an interactive prompt-based chat UI with **Gradio** for ease of experimentation.
- Support Thai-language input/output with real-time responses.

### 🎢 Processes

1. **Model Loading** – Download and initialize OpenThaiGPT model weights.
2. **Prompt Handling** – Manage input queries and generate model responses.
3. **UI Development** – Implement Gradio interface for user interaction.
4. **Evaluation & Testing** – Validate responsiveness and output relevance.

### 🎯 Results & Impact

- Provides a functional baseline for Thai-language LLM interaction.
- Enables researchers and developers to experiment with Thai NLP applications.
- Demonstrates effective local deployment of large language models with user-friendly interface.

### ⚙️ Development Challenges

- Efficiently loading large models in resource-constrained environments.
- Ensuring smooth user experience with minimal latency.
- Handling Thai language nuances in generation and understanding.


## **Key Features**:
  - Loads and runs **OpenThaiGPT-7B-chat** (Hugging Face model) using local inference tools
  - Supports **prompt-based interaction** with Thai language queries
  - Simple and responsive **Gradio UI** for interactive chat interface
  - Lays the foundation for Thai-language applications like virtual assistants, summarizers, or translators

## **Prototype Scope**:
  - Early-stage prototype focusing on functionality rather than optimization
  - Intended as a learning and experimentation platform for Thai LLM integration

## **Libraries Used**:
  - **Data Handling**: `pandas`, `NumPy`
  - **Web Interface**: `gradio`
  - **LLM & Backend**:
    - `pytorch`
    - `llama-index` (for managing documents/chunks, if needed)
    - `llama-cpp-python` (for lightweight LLM execution)

## **Deliverables**:
  - Demo Notebook: [`DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb`](DL_5_WITH_openthaigpt_1_0_0_7b_chat.ipynb)

---
