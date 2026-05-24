import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_PRED_PATH = Path("Model_Finetune/outputs/predictions/base_led_predictions_370.jsonl")
FINETUNED_PRED_PATH = Path("Model_Finetune/outputs/predictions/finetuned_led_predictions_370.jsonl")

OUTPUT_DIR = Path("Model_Finetune/outputs/evaluation")
RESULTS_CSV = OUTPUT_DIR / "evaluation_results_370.csv"
SUMMARY_CSV = OUTPUT_DIR / "evaluation_summary_370.csv"

# Local sentence-transformer model copied to Wolffe.
# This prevents Wolffe from trying to download from Hugging Face.
HSSM_MODEL_NAME = "Model_Finetune/models/all-mpnet-base-v2"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def word_count(text):
    if not text:
        return 0

    return len(text.split())


def split_sentences(text):
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    return sentences


def compute_hssm(candidate, reference, embedding_model):
    candidate_sentences = split_sentences(candidate)
    reference_sentences = split_sentences(reference)

    if not candidate_sentences or not reference_sentences:
        return 0.0

    candidate_embeddings = embedding_model.encode(
        candidate_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    reference_embeddings = embedding_model.encode(
        reference_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity_matrix = cosine_similarity(candidate_embeddings, reference_embeddings)

    # Hungarian algorithm minimizes cost, so we use negative similarity as cost.
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

    matched_scores = similarity_matrix[row_indices, col_indices]

    if len(matched_scores) == 0:
        return 0.0

    return float(np.mean(matched_scores))


def evaluate_model(records, model_label, rouge, embedding_model, bert_device):
    rows = []

    candidates = [record["generated_summary"] for record in records]
    references = [record["reference_summary"] for record in records]

    print(f"Computing BERTScore for {model_label}...")

    # Use local model path to avoid Hugging Face download on Wolffe.
    _, _, bert_f1_scores = bert_score(
        candidates,
        references,
        model_type=HSSM_MODEL_NAME,
        device=bert_device,
        verbose=True,
        rescale_with_baseline=False,
    )

    print(f"Computing ROUGE-L and HSSM for {model_label}...")

    for idx, record in enumerate(records):
        generated = record["generated_summary"]
        reference = record["reference_summary"]

        rouge_scores = rouge.score(reference, generated)
        rouge_l = rouge_scores["rougeL"].fmeasure

        hssm = compute_hssm(
            candidate=generated,
            reference=reference,
            embedding_model=embedding_model,
        )

        row = {
            "arxiv_id": record["arxiv_id"],
            "title": record["title"],
            "model": model_label,
            "rouge_l": rouge_l,
            "bertscore_f1": float(bert_f1_scores[idx]),
            "hssm": hssm,
            "generated_word_count": word_count(generated),
            "reference_word_count": word_count(reference),
        }

        rows.append(row)

    return rows


def make_summary_table(results_df):
    summary = (
        results_df
        .groupby("model")
        .agg(
            rouge_l_mean=("rouge_l", "mean"),
            rouge_l_std=("rouge_l", "std"),
            bertscore_f1_mean=("bertscore_f1", "mean"),
            bertscore_f1_std=("bertscore_f1", "std"),
            hssm_mean=("hssm", "mean"),
            hssm_std=("hssm", "std"),
            generated_word_count_mean=("generated_word_count", "mean"),
            generated_word_count_std=("generated_word_count", "std"),
            reference_word_count_mean=("reference_word_count", "mean"),
        )
        .reset_index()
    )

    return summary


def plot_average_metrics(summary_df):
    metric_columns = ["rouge_l_mean", "bertscore_f1_mean", "hssm_mean"]
    metric_labels = ["ROUGE-L", "BERTScore F1", "HSSM"]

    x = np.arange(len(metric_labels))
    width = 0.35

    base_values = summary_df[summary_df["model"] == "base_led"][metric_columns].values[0]
    finetuned_values = summary_df[summary_df["model"] == "led_qlora_5000"][metric_columns].values[0]

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, base_values, width, label="Base LED")
    plt.bar(x + width / 2, finetuned_values, width, label="LED + QLoRA")

    plt.xticks(x, metric_labels)
    plt.ylabel("Average score")
    plt.title("Average Evaluation Metrics: Base LED vs LED + QLoRA")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "average_metric_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_metric_distributions(results_df):
    metrics = ["rouge_l", "bertscore_f1", "hssm"]
    labels = ["ROUGE-L", "BERTScore F1", "HSSM"]

    data = []
    positions = []
    tick_labels = []

    pos = 1

    for metric, label in zip(metrics, labels):
        base_values = results_df[results_df["model"] == "base_led"][metric].values
        finetuned_values = results_df[results_df["model"] == "led_qlora_5000"][metric].values

        data.extend([base_values, finetuned_values])
        positions.extend([pos, pos + 1])
        tick_labels.extend([f"{label}\nBase", f"{label}\nQLoRA"])

        pos += 3

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, positions=positions, widths=0.6, showmeans=True)
    plt.xticks(positions, tick_labels)
    plt.ylabel("Score")
    plt.title("Metric Score Distributions")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "metric_distribution_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_summary_lengths(results_df):
    base_lengths = results_df[results_df["model"] == "base_led"]["generated_word_count"].values
    finetuned_lengths = results_df[results_df["model"] == "led_qlora_5000"]["generated_word_count"].values

    # Reference summaries are the same for both models, so use only one copy.
    reference_lengths = (
        results_df[results_df["model"] == "base_led"]["reference_word_count"].values
    )

    data = [base_lengths, finetuned_lengths, reference_lengths]
    labels = ["Base LED", "LED + QLoRA", "Reference Abstract"]

    plt.figure(figsize=(9, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Word count")
    plt.title("Generated Summary Length Compared with Reference Abstracts")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "summary_length_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prediction files...")
    base_records = load_jsonl(BASE_PRED_PATH)
    finetuned_records = load_jsonl(FINETUNED_PRED_PATH)

    print(f"Loaded base predictions: {len(base_records)}")
    print(f"Loaded fine-tuned predictions: {len(finetuned_records)}")

    if len(base_records) != len(finetuned_records):
        raise ValueError("Base and fine-tuned prediction files have different lengths.")

    bert_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using BERTScore/HSSM device: {bert_device}")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print(f"Loading local HSSM sentence embedding model: {HSSM_MODEL_NAME}")
    embedding_model = SentenceTransformer(HSSM_MODEL_NAME, device=bert_device)

    all_rows = []

    all_rows.extend(
        evaluate_model(
            records=base_records,
            model_label="base_led",
            rouge=rouge,
            embedding_model=embedding_model,
            bert_device=bert_device,
        )
    )

    all_rows.extend(
        evaluate_model(
            records=finetuned_records,
            model_label="led_qlora_5000",
            rouge=rouge,
            embedding_model=embedding_model,
            bert_device=bert_device,
        )
    )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"Saved detailed results to: {RESULTS_CSV}")

    summary_df = make_summary_table(results_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved summary results to: {SUMMARY_CSV}")
    print("\nSummary:")
    print(summary_df)

    plot_average_metrics(summary_df)
    plot_metric_distributions(results_df)
    plot_summary_lengths(results_df)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()