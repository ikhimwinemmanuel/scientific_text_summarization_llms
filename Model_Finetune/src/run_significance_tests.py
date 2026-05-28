from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


EVAL_RESULTS_PATH = Path("Model_Finetune/outputs/evaluation/evaluation_results_370.csv")
OUTPUT_PATH = Path("Model_Finetune/outputs/evaluation/significance_tests_370.csv")


METRICS = {
    "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "hssm": "HSSM",
}


def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return "Statistically significant"
    return "Not statistically significant"


def main():
    df = pd.read_csv(EVAL_RESULTS_PATH)

    base_df = df[df["model"] == "base_led"].copy()
    fine_df = df[df["model"] == "led_qlora_5000"].copy()

    base_df = base_df.sort_values("arxiv_id").reset_index(drop=True)
    fine_df = fine_df.sort_values("arxiv_id").reset_index(drop=True)

    if len(base_df) != len(fine_df):
        raise ValueError("Base and fine-tuned result files do not have the same number of records.")

    if not all(base_df["arxiv_id"] == fine_df["arxiv_id"]):
        raise ValueError("Base and fine-tuned records are not aligned by arXiv ID.")

    rows = []

    for metric_col, metric_name in METRICS.items():
        base_scores = base_df[metric_col]
        fine_scores = fine_df[metric_col]

        differences = fine_scores - base_scores

        statistic, p_value = wilcoxon(
            fine_scores,
            base_scores,
            alternative="greater"
        )

        rows.append(
            {
                "metric": metric_name,
                "base_mean": base_scores.mean(),
                "finetuned_mean": fine_scores.mean(),
                "mean_difference": differences.mean(),
                "median_difference": differences.median(),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "significance_at_0.05": interpret_p_value(p_value),
            }
        )

    results_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(results_df)
    print(f"\nSaved significance test results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()