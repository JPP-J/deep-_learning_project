# Chat with LLM Projects Project 💬
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :


## 📌 Overview

This project provides an interactive chat application prototype utilizing the **Qwen** and **Qwen2.5** Large Language Models (LLMs). Designed to run efficiently with GPU acceleration (`torch-cuda`), it demonstrates key capabilities for building conversational AI systems that maintain multi-turn chat history and generate context-aware responses.

### 🧩 Problem Statement

Building conversational agents that can engage users in natural, coherent multi-turn dialogue requires preserving conversation context and generating relevant replies. This prototype aims to showcase how to implement such a system using state-of-the-art LLMs.

### 🔍 Approach

- Use pretrained Qwen and Qwen2.5 models accessed through the `transformers` library.
- Leverage GPU acceleration for faster inference and response generation.
- Implement reusable functions to send and receive messages, managing chat history for contextual continuity.
- Enable experimentation with conversational AI by providing a simple, extensible framework.

### 🎢 Processes

1. Initialize models with GPU support.
2. Implement chat history management for multi-turn dialogue.
3. Develop example message handling and response generation functions.
4. Test interactive chat scenarios to validate context retention.
5. Optimize response times via hardware acceleration.

### 🎯 Results & Impact

- Demonstrates effective multi-turn chat functionality with large language models.
- Enables development and experimentation with conversational AI applications.
- Provides a reusable template for further customization and integration.

### ⚙️ Challenges and Considerations

- **Context Management:** Maintaining accurate and efficient multi-turn chat history can be complex, especially as conversation length grows.
- **Latency:** Generating responses in real-time requires optimization of model inference time, which depends on hardware capabilities (GPU acceleration is essential).
- **Memory Usage:** Large LLMs consume significant memory, limiting batch size and number of concurrent users.
- **Response Quality:** Balancing coherence, relevance, and creativity in responses needs careful tuning of model parameters and prompt design.
- **Error Handling:** Managing unexpected inputs or ambiguous queries while preserving conversation flow is non-trivial.
- **Scalability:** Extending the prototype for production-level systems involves integrating user authentication, persistent storage, and deployment considerations.

## **Prototype Scope**:  
  This prototype demonstrates how to build an interactive chat system using large language models (LLMs) with preserved message history. It simulates human-like conversations by maintaining context between turns and generating context-aware replies.

## **Libraries Used**:
  - **Processing:** `torch-cuda`
  - **Deep Learning/AI:** `transformers`

## **Models**: Qwen, Qwen2.5

## **Features**:
  - Preserves multi-turn chat history for continuity
  - Example functions to send/receive messages
  - Supports conversational AI experimentation

## **Deliverables**:
  - Hands-on Demo Notebook: [`DL_8_chat_LLM.ipynb`](DL_8_chat_LLM.ipynb)

---
