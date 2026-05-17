import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "ccdv/arxiv-summarization"
SPLIT = "train"
NUM_SAMPLES = 5000
SEED = 42

OUTPUT_PATH = Path("Model_Finetune/data/processed/train_arxiv_5000.jsonl")


def clean_text(text):
    """
    Basic whitespace cleaning only.
    We avoid heavy cleaning because scientific terminology and structure
    should be preserved for summarisation fine-tuning.
    """
    if text is None:
        return ""
    return " ".join(text.split())


def save_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    total_records = len(dataset)
    print(f"Total records available: {total_records}")

    if NUM_SAMPLES > total_records:
        raise ValueError(
            f"NUM_SAMPLES={NUM_SAMPLES} is greater than dataset size={total_records}"
        )

    random.seed(SEED)
    selected_indices = random.sample(range(total_records), NUM_SAMPLES)

    records = []

    for new_id, idx in enumerate(tqdm(selected_indices, desc="Preparing training records")):
        row = dataset[idx]

        article = clean_text(row["article"])
        abstract = clean_text(row["abstract"])

        if not article or not abstract:
            continue

        record = {
            "id": new_id,
            "source_dataset": DATASET_NAME,
            "input_text": article,
            "target_summary": abstract
        }

        records.append(record)

    save_jsonl(records, OUTPUT_PATH)

    print(f"\nSaved {len(records)} records")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()