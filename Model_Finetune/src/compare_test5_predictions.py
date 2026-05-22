import json
from pathlib import Path


BASE_PATH = Path("Model_Finetune/outputs/predictions/base_led_predictions_test5.jsonl")
FINETUNED_PATH = Path("Model_Finetune/outputs/predictions/finetuned_led_predictions_test5.jsonl")
OUTPUT_PATH = Path("Model_Finetune/outputs/predictions/comparison_test5.txt")


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():
    base_records = load_jsonl(BASE_PATH)
    finetuned_records = load_jsonl(FINETUNED_PATH)

    base_by_id = {record["arxiv_id"]: record for record in base_records}
    finetuned_by_id = {record["arxiv_id"]: record for record in finetuned_records}

    common_ids = list(base_by_id.keys())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, arxiv_id in enumerate(common_ids, start=1):
            base = base_by_id[arxiv_id]
            fine = finetuned_by_id[arxiv_id]

            f.write("=" * 100 + "\n")
            f.write(f"Record {idx}\n")
            f.write(f"ArXiv ID: {arxiv_id}\n")
            f.write(f"Title: {base['title']}\n\n")

            f.write("BASE LED GENERATED SUMMARY:\n")
            f.write(base["generated_summary"].strip() + "\n\n")

            f.write("FINE-TUNED LED GENERATED SUMMARY:\n")
            f.write(fine["generated_summary"].strip() + "\n\n")

            f.write("REFERENCE ABSTRACT:\n")
            f.write(base["reference_summary"].strip() + "\n\n")

    print(f"Saved comparison file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()