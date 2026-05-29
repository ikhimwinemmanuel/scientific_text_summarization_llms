from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


EVAL_RESULTS_PATH = Path("Model_Finetune/outputs/evaluation/evaluation_results_370.csv")
OUTPUT_PATH = Path("Model_Finetune/outputs/evaluation/metric_correlations_370.csv")


METRICS = {
    "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "hssm": "HSSM",
}


def compute_correlations(df, model_name):
    rows = []

    metric_pairs = [
        ("hssm", "rouge_l"),
        ("hssm", "bertscore_f1"),
        ("rouge_l", "bertscore_f1"),
    ]

    for metric_a, metric_b in metric_pairs:
        pearson_corr, pearson_p = pearsonr(df[metric_a], df[metric_b])
        spearman_corr, spearman_p = spearmanr(df[metric_a], df[metric_b])

        rows.append(
            {
                "model": model_name,
                "metric_pair": f"{METRICS[metric_a]} vs {METRICS[metric_b]}",
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
            }
        )

    return rows


def main():
    print("Loading evaluation results...")
    df = pd.read_csv(EVAL_RESULTS_PATH)

    all_rows = []

    for model_name in ["base_led", "led_qlora_5000"]:
        model_df = df[df["model"] == model_name].copy()

        if len(model_df) == 0:
            raise ValueError(f"No records found for model: {model_name}")

        print(f"Computing correlations for {model_name}: {len(model_df)} records")
        all_rows.extend(compute_correlations(model_df, model_name))

    correlation_df = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    correlation_df.to_csv(OUTPUT_PATH, index=False)

    print("\nCorrelation results:")
    print(correlation_df)

    print(f"\nSaved correlation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()