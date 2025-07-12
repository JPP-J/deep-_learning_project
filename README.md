# Generative and Summarization Tasks Using LLMs Project ✍️📄
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

### 📌 Overview

This project showcases the use of popular pretrained Large Language Models (LLMs) to perform two fundamental NLP tasks: **text generation** and **text summarization**. By leveraging models like **GPT-2** and **facebook/bart-large-cnn**, the prototype demonstrates how to generate coherent text from prompts and produce concise summaries from longer documents.

#### 🧩 Problem Statement

Generating human-like text and summarizing long documents are critical tasks in NLP with broad applications such as content creation, information retrieval, and conversational AI. This project aims to provide a simple, accessible demonstration of these capabilities using state-of-the-art pretrained models.

#### 🔍 Approach

- Utilize `transformers` library for easy access to pretrained LLMs.
- Implement `generate_text(prompt)` for creative or context-aware text completions with controllable decoding strategies.
- Implement `summarize_text(input_text)` to extract and condense key information.
- Support experimentation with decoding techniques like top-k and top-p sampling to balance creativity and relevance.

#### 🎢 Processes

1. Load pretrained GPT-2 for text generation.
2. Load pretrained BART large CNN model for summarization.
3. Define helper functions for generation and summarization.
4. Experiment with different decoding parameters.
5. Evaluate output quality on example inputs.

#### 🎯 Results & Impact

- Demonstrates effective generation of fluent, coherent text based on user prompts.
- Produces informative and concise summaries suitable for downstream tasks.
- Serves as a foundation for applications such as chatbots, content summarization tools, and automated writing assistants.

#### ⚙️ Development Challenges

- Balancing creativity vs. factual accuracy in generated text.
- Managing length and informativeness in summaries.
- Efficiently running large models within resource constraints.


## **Key Features**:
  - Implements two core NLP tasks using pretrained models via the `transformers` library
  - Includes example functions for:
    - `generate_text(prompt)` – Generate creative or context-aware completions
    - `summarize_text(input_text)` – Extract key information in a shorter form
  - Supports experimentation with different decoding methods (e.g., top-k, top-p sampling)

## **Prototype Scope**:
  - Standalone notebook demonstrating usage
  - Ideal for content generation, article summarization, or chatbot backend

## **Libraries Used**:
  - **Model Access & Execution**: `pytorch`, `transformers`
  - **Models**:
    - `gpt2` (text generation)
    - `facebook/bart-large-cnn` (summarization)

## **Deliverables**:
  - Demo Notebook: [`DL_6_summarize_gen_text.ipynb`](DL_6_summarize_gen_text.ipynb)
---
