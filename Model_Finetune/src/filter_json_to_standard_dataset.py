import json
from pathlib import Path

INPUT = Path("Model_Finetune/data/processed/final_dataset.jsonl")
OUTPUT = Path("Model_Finetune/data/processed/final_dataset_clean.jsonl")


def word_count(text):
    return len(text.split()) if text else 0


def is_valid(r):
    return (
        word_count(r["abstract"]) >= 80 and
        word_count(r["introduction"]) >= 150 and
        word_count(r["conclusion"]) >= 80
    )


def main():
    kept = 0
    dropped = 0

    with open(INPUT, "r", encoding="utf-8") as f_in, \
         open(OUTPUT, "w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)

            if is_valid(record):
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()