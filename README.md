# Prompt-based Generation with LLM Project ✨
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :


## 📌 Overview

This project demonstrates various techniques for generating coherent and context-aware text using prompt-based approaches with large language models (LLMs) like Qwen and Qwen2.5.

### 🧩 Problem Statement

Generating high-quality text outputs that align well with user prompts is challenging due to the complexity of natural language and the nuances of prompt design. Users often require flexible methods to experiment with different prompting strategies to optimize generation results.

### 🔍 Approach

The prototype explores both direct model/tokenizer usage and the high-level `transformers.pipeline()` API to provide a comprehensive toolkit for prompt-based text generation. It emphasizes modular code and example workflows, enabling experimentation with diverse prompting styles and decoding techniques.

### 🎢 Processes

1. **Model Loading** – Initialize Qwen/Qwen2.5 models and tokenizers either directly or through pipelines  
2. **Prompt Design** – Craft various prompt formats to test generation behaviors  
3. **Text Generation Methods** – Implement multiple approaches including greedy decoding, sampling, and beam search  
4. **Generation Tuning** – Adjust parameters such as max tokens, temperature, and top-k/top-p sampling  
5. **Evaluation & Examples** – Provide usage demos and output inspection for different prompts and methods  

### 🎯 Results & Impact

- Demonstrated effective generation of contextually relevant text from prompts  
- Enabled rapid prototyping and testing of prompt strategies  
- Facilitated understanding of LLM behavior in generation tasks  

### ⚙️ Challenges and Considerations

- Managing token and context window limits for longer or complex prompts  
- Balancing generation quality against latency and computational cost  
- Designing robust prompts to reduce unwanted bias or repetitive outputs  
- Handling edge cases and ambiguous inputs gracefully  


## **Prototype Scope**:  
  This prototype explores how to generate coherent text using various prompt-based techniques with Qwen/Qwen2.5 LLMs. It demonstrates direct and abstracted model usage to support flexible experimentation in language generation tasks.

## **Libraries Used**:
  - **Processing:** `torch-cuda`
  - **Deep Learning/AI:** `transformers`

## **Models**: Qwen, Qwen2.5

##- **Features**:
  - Demonstrates direct and high-level generation methods
  - Covers best practices in prompt design
  - Includes usage examples for each method

## **Deliverables**:
  - Hands-on Demo Notebook: [`DL_9_Prompt_based_Generation.ipynb`](DL_9_Prompt_based_Generation.ipynb)
---
