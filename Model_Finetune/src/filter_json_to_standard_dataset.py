import json
import re
from pathlib import Path

INPUT = Path("Model_Finetune/data/processed/final_dataset.jsonl")
OUTPUT = Path("Model_Finetune/data/processed/final_dataset_clean.jsonl")


def word_count(text):
    return len(text.split()) if text else 0


def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+|doi\.org/\S+', '', text, flags=re.IGNORECASE)


def remove_inline_citations(text):
    # Removes patterns like [1], [2], [1, 2], [12,13,14]
    return re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)


def truncate_at_references(text):
    """
    Cuts off text once a likely references/bibliography section starts.
    """
    if not text:
        return text

    patterns = [
        r'\n\s*references\s*\n',
        r'\n\s*bibliography\s*\n',
        r'\n\s*acknowledg(?:e)?ments?\s*\n'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()].strip()

    return text.strip()


def normalize_text(text):
    if not text:
        return ""

    text = remove_urls(text)
    text = remove_inline_citations(text)
    text = truncate_at_references(text)

    # remove repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def has_table_artifacts(text):
    if not text:
        return False

    table_patterns = [
        r'\|.*\|',        # markdown-style table rows
        r'\|---',         # markdown separator
    ]

    return any(re.search(p, text) for p in table_patterns)


def url_count(text):
    if not text:
        return 0
    return len(re.findall(r'https?://\S+|www\.\S+|doi\.org/\S+', text, flags=re.IGNORECASE))


def is_valid(r):
    abstract = r.get("abstract", "")
    introduction = r.get("introduction", "")
    conclusion = r.get("conclusion", "")

    # minimum thresholds (your original idea)
    if word_count(abstract) < 80:
        return False
    if word_count(introduction) < 150:
        return False
    if word_count(conclusion) < 80:
        return False

    # maximum thresholds (to remove over-extracted sections)
    if word_count(abstract) > 500:
        return False
    if word_count(introduction) > 2500:
        return False
    if word_count(conclusion) > 1800:
        return False

    # obvious formatting artifacts
    if has_table_artifacts(abstract) or has_table_artifacts(introduction) or has_table_artifacts(conclusion):
        return False

    # too many URLs usually means noisy extraction
    if url_count(abstract) > 2 or url_count(introduction) > 2 or url_count(conclusion) > 2:
        return False

    return True


def main():
    kept = 0
    dropped = 0

    with open(INPUT, "r", encoding="utf-8") as f_in, \
         open(OUTPUT, "w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)

            # clean text first
            record["abstract"] = normalize_text(record.get("abstract", ""))
            record["introduction"] = normalize_text(record.get("introduction", ""))
            record["conclusion"] = normalize_text(record.get("conclusion", ""))

            if is_valid(record):
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()