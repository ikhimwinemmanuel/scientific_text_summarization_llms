# Leveraging Large Language Models for Scientific Text Summarization: Fine-Tuning LED and Performance Evaluation 


## Project Overview

This repository contains the implementation for an INFO7016 Postgraduate Project B research project on scientific text summarization using the Longformer Encoder-Decoder (LED) model. The project evaluates whether parameter-efficient fine-tuning using Quantized Low-Rank Adaptation (QLoRA) improves the summarization performance of a base LED model on scientific papers.

The project compares two model settings:

1. Base LED model without fine-tuning
2. QLoRA fine-tuned LED model

Both models are evaluated on the same fixed test dataset of 370 cleaned arXiv papers. The generated summaries are compared against the original abstracts using ROUGE-L, BERTScore F1, and a proposed sentence-level evaluation metric called the Hungarian Summary Similarity Metric (HSSM).

## Research Aim

The main aim of this project is to evaluate whether QLoRA fine-tuning improves the scientific summarization performance of a base LED model.

The project also proposes HSSM as a complementary sentence-level evaluation metric for comparing generated summaries with reference abstracts.

## Main Contributions

This project makes three main contributions:

1. A controlled before-and-after comparison of a base LED model and a QLoRA fine-tuned LED model for scientific text summarization.
2. The implementation of the Hungarian Summary Similarity Metric (HSSM), a sentence-level metric based on sentence embeddings, cosine similarity, and Hungarian one-to-one alignment.
3. A multi-metric evaluation framework using ROUGE-L, BERTScore F1, HSSM, Wilcoxon signed-rank significance testing, and inter-metric correlation analysis.

## Final Results Summary

The QLoRA fine-tuned LED model achieved higher average scores than the base LED model across all three evaluation metrics.

| Metric       | Base LED | LED + QLoRA |
| ------------ | -------: | ----------: |
| ROUGE-L      |   0.1525 |      0.1780 |
| BERTScore F1 |   0.6831 |      0.7172 |
| HSSM         |   0.5454 |      0.5820 |

A paired Wilcoxon signed-rank test showed that the improvements were statistically significant across ROUGE-L, BERTScore F1, and HSSM.

## Repository Structure

```text
scientific_text_summarization_llms/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Model_Finetune/
│   └── src/
│       ├── download_arxiv_papers.py
│       ├── download_fixed_papers.py
│       ├── create_fixed_paper_list.py
│       ├── pymupdf4llm_extraction.py
│       ├── filter_json_to_standard_dataset.py
│       ├── build_test_dataset.py
│       ├── prepare_train_data.py
│       ├── download_led_base_model.py
│       ├── train_led_qlora_test.py
│       ├── train_led_qlora_full.py
│       ├── run_base_led_inference_test.py
│       ├── run_base_led_inference_full.py
│       ├── run_finetuned_led_inference_test.py
│       ├── run_finetuned_led_inference_full.py
│       ├── evaluate_predictions.py
│       ├── run_significance_tests.py
│       ├── analyse_metric_correlations.py
│       └── build_evaluation_table.py
│
└── hss_metric/
    ├── README.md
    └── src/
        └── hss_metric.py
```

## Important Note on Large Files

The following Large files are intentionally excluded from this GitHub repository using `.gitignore`. :

* downloaded arXiv PDFs
* processed datasets
* Hugging Face model files
* trained QLoRA adapters
* generated prediction files
* evaluation outputs and charts
* virtual environment files

## Methodology Summary

The project follows a controlled before-and-after experimental design.

First, the base LED model generates summaries for the fixed 370-paper test dataset created for this project.The 370-paper test dataset was created by selecting fixed arXiv papers, downloading their PDFs, extracting the title, introduction, conclusion, and abstract, and then cleaning the extracted text for evaluation.Then, the same base LED model is fine-tuned using QLoRA on a separate 5,000-record arXiv summarization training dataset sourced from the Hugging Face [`ccdv/arxiv-summarization`](https://huggingface.co/datasets/ccdv/arxiv-summarization) dataset. After fine-tuning, the QLoRA-adapted LED model generates summaries for the same 370 test papers.


Both models use the same input format and generation settings to ensure a fair comparison.

The evaluation input consists of:

```text
title + introduction + conclusion
```

The original abstract is held out and used as the reference summary.

## Model and Fine-Tuning

The base model used in this project is:

```text
allenai/led-base-16384
```

QLoRA was used for parameter-efficient fine-tuning. During fine-tuning, only 589,824 parameters were trainable out of 162,434,304 total parameters, representing approximately 0.3631% of the model parameters.

## Inference Settings

Both the base LED model and the QLoRA fine-tuned LED model used the same decoding settings:

```text
maximum input length: 4096 tokens
beam size: 4
minimum summary length: 50 tokens
maximum summary length: 180 tokens
no_repeat_ngram_size: 4
```

## Evaluation Metrics

Three evaluation metrics were used:

### ROUGE-L

ROUGE-L measures lexical overlap using the longest common subsequence between the generated summary and the reference abstract.

### BERTScore F1

BERTScore measures semantic similarity using contextual token embeddings.

### HSSM

HSSM is a proposed sentence-level evaluation metric. It works by:

1. Splitting the generated summary and reference abstract into sentences
2. Encoding each sentence using sentence-transformer embeddings
3. Computing cosine similarity between generated-summary sentences and reference-abstract sentences
4. Applying the Hungarian algorithm to find the best one-to-one sentence alignment optimal match.
5. Averaging the cosine similarity scores of the matched sentence pairs

HSSM is not intended to replace already existing ROUGE-L or BERTScore. It is used as a complementary metric for analysing sentence-level semantic alignment.

## Reproducibility Guide

### 1. Clone the repository

```bash
git clone https://github.com/ikhimwinemmanuel/scientific_text_summarization_llms.git
cd scientific_text_summarization_llms
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

For Linux or HPC environments:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data

The data preparation scripts are located in:

```text
Model_Finetune/src/
```

The main scripts used for data preparation include:

```bash
python Model_Finetune/src/download_arxiv_papers.py
python Model_Finetune/src/pymupdf4llm_extraction.py
python Model_Finetune/src/filter_json_to_standard_dataset.py
python Model_Finetune/src/build_test_dataset.py
python Model_Finetune/src/prepare_train_data.py
```

### 5. Download or prepare the base LED model

```bash
python Model_Finetune/src/download_led_base_model.py
```

### 6. Run a small QLoRA training test

```bash
python Model_Finetune/src/train_led_qlora_test.py
```

### 7. Run full QLoRA fine-tuning

```bash
python Model_Finetune/src/train_led_qlora_full.py
```

### 8. Run inference with the base LED model

```bash
python Model_Finetune/src/run_base_led_inference_full.py
```

### 9. Run inference with the QLoRA fine-tuned LED model

```bash
python Model_Finetune/src/run_finetuned_led_inference_full.py
```

### 10. Evaluate predictions

```bash
python Model_Finetune/src/evaluate_predictions.py
python Model_Finetune/src/run_significance_tests.py
python Model_Finetune/src/analyse_metric_correlations.py
python Model_Finetune/src/build_evaluation_table.py
```

## Hardware Environment

The full experiments were run on Western Sydney University Wolffe High Performance Computing environment using an NVIDIA A30 GPU through the `ampere24` partition. GPU jobs were submitted through Slurm Scheduler.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* bitsandbytes
* Sentence-Transformers
* SciPy
* scikit-learn
* pandas
* NumPy
* matplotlib
* rouge-score
* bert-score
* Git and GitHub
* Slurm
* Wolffe HPC

## Project Status

The project implementation is complete. The repository contains the main scripts used for data preparation, QLoRA fine-tuning, inference, evaluation, statistical significance testing, and HSSM analysis.

## Author

- Ikhimwin Osakpamwan Emmanuel
- Master of Artificial Intelligence
- Western Sydney University