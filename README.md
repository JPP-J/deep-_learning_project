# Deep learning Projects 🤖
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/deep-_learning_project?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/deep-_learning_project?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/JPP-J/deep-_learning_project?style=flat-square)


This repo is home to the code that accompanies Jidapa's *Deep Learining Project* :

## Generative and Summarization Tasks Using LLMs Project ✍️📄

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
