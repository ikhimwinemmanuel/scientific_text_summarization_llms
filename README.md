# Leveraging Large Language Models for Scientific Text Summarization

## Overview

This project explores the use of Large Language Models (LLMs) for scientific text summarization, with a focus on both model performance and evaluation methods.

It builds on a previous Project(A) where the benchmarking of multiple transformer-based models from huggingface were evaluated on scientific papers from Arxiv. In this stage Project(B), the focus shifts towards improving model performance and developing a more robust evaluation metric.

---

## Project Objectives

* Improve summarization performance through fine-tuning (LED model)
* Compare model performance before and after fine-tuning
* Evaluate summaries using:

  * ROUGE
  * BERTScore
  * A proposed metric: **HSSM (Hungarian Summary Similarity Metric)**

---

## Background (Project A)

In earlier work, three Hugging Face models were benchmarked:

* T5
* PEGASUS
* LED

These models were applied to scientific papers (arXiv) and evaluated using:

* ROUGE (lexical overlap)
* BERTScore (semantic similarity)
* Computational efficiency (runtime and resource usage)

From this benchmarking, **LED** was identified as the best-performing model.

---

## Motivation

Existing evaluation metrics have limitations:

* **ROUGE** focuses on surface-level word overlap
* **BERTScore** captures semantic similarity but operates at the token level using greedy matching

These approaches:

* Do not capture sentence-level structure
* May overestimate similarity due to repeated matching

This motivates the need for a more structured evaluation method.

---

## Proposed Method: HSSM

The **Hungarian Summary Similarity Metric (HSSM)** is a sentence-level evaluation approach designed to better capture structural and semantic alignment between summaries.

### Core Idea

1. Split summaries into sentences
2. Generate sentence embeddings
3. Compute a cosine similarity matrix
4. Apply Hungarian matching to enforce optimal one-to-one alignment
5. Compute precision, recall, and F1 score

This approach ensures that each sentence is matched uniquely, improving the reliability of similarity evaluation.

---

## Current Progress

* Repository structure established
* Benchmarking pipeline (Project A) completed
* Dataset collection and preprocessing implemented
* HSSM design defined
* Initial implementation of HSSM pipeline in progress

---

## Next Steps

* Fine-tune the LED model on scientific text data
* Run full evaluation across:

  * ROUGE
  * BERTScore
  * HSSM
* Analyse differences between metrics
* Validate the effectiveness of HSSM

---

## Technologies Used

* Python
* Hugging Face Transformers
* PyTorch
* Sentence-Transformers (`all-mpnet-base-v2`)
* SciPy (Hungarian algorithm)
* Evaluation libraries (ROUGE, BERTScore)

---

## Repository Structure

```
scientific_text_summarization_llms/
├── hss_metric/
│   ├── data/
│   ├── outputs/
│   ├── src/
│   │   └── hss_metric.py
│   └── README.md
│
├── model_finetune/
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── outputs/
│   └── src/
│       ├── download_arxiv_papers.py
│       └── test_run.py
│
├── .gitignore
├── README.md
└── requirements.txt

```

---

## Status

🚧 This project is currently in progress as part of a Master’s level project 

---

## Author

Emmanuel Ikhimwin
Master of Artificial Intelligence
Western Sydney University
