import json
from pathlib import Path


INPUT_PATH = Path("Model_Finetune/data/processed/final_dataset_cleanv2.jsonl")
OUTPUT_PATH = Path("Model_Finetune/data/processed/test_dataset_370_intro_conclusion.jsonl")


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_input_text(title, introduction, conclusion):
    return (
        f"Title:\n{title.strip()}\n\n"
        f"Introduction:\n{introduction.strip()}\n\n"
        f"Conclusion:\n{conclusion.strip()}"
    )


def main():
    records = load_jsonl(INPUT_PATH)
    output_records = []

    for record in records:
        arxiv_id = record.get("arxiv_id", "").strip()
        title = record.get("title", "").strip()
        abstract = record.get("abstract", "").strip()
        introduction = record.get("introduction", "").strip()
        conclusion = record.get("conclusion", "").strip()

        if not arxiv_id or not title or not abstract or not introduction or not conclusion:
            continue

        output_record = {
            "arxiv_id": arxiv_id,
            "title": title,
            "input_text": build_input_text(title, introduction, conclusion),
            "reference_summary": abstract,
            "input_source": "title_introduction_conclusion",
        }

        output_records.append(output_record)

    save_jsonl(output_records, OUTPUT_PATH)

    print(f"Loaded records: {len(records)}")
    print(f"Saved evaluation records: {len(output_records)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()