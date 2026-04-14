import json
import re
from pathlib import Path
import pymupdf4llm
from tqdm import tqdm


INPUT_PATH = Path("model_finetune/data/processed/fixed_papers.jsonl")
PDF_DIR = Path("model_finetune/data/raw/pdfs")
OUTPUT_PATH = Path("model_finetune/data/processed/final_dataset.jsonl")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_section_text(text):
    if not text:
        return None

    text = re.sub(r"\*\*==> picture.*?<==\*\*", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"\bFigure\s+\d+.*?(?=\bFigure\s+\d+\b|$)",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"\bTable\s+\d+.*?(?=\bTable\s+\d+\b|$)",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"[*`#<>_]", " ", text)
    text = re.sub(r"\b\d+\s*br\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    return normalize_text(text)


def extract_abstract(md_text):
    text = normalize_text(md_text)

    explicit_pattern = re.compile(
        r"\babstract\b[:\s]*(.*?)(?=\b(?:1\.?\s*introduction|introduction)\b)",
        re.IGNORECASE
    )
    match = explicit_pattern.search(text)
    if match:
        return clean_section_text(match.group(1).strip())

    intro_match = re.search(r"\b(?:1\.?\s*introduction|introduction)\b", text, re.IGNORECASE)
    if not intro_match:
        return None

    before_intro = text[:intro_match.start()].strip()

    split_markers = [
        r"\bcorrespondence\b.*?(?=\bdate\b|$)",
        r"\bdate\b.*?$",
        r"\bproject page\b.*?$",
        r"\bcode\b.*?$"
    ]

    candidate = before_intro
    for marker in split_markers:
        candidate = re.sub(marker, " ", candidate, flags=re.IGNORECASE)

    candidate = clean_section_text(candidate)

    sentences = re.split(r"(?<=[.!?])\s+", candidate)
    filtered = [s for s in sentences if len(s.split()) > 8]

    if len(filtered) >= 3:
        return " ".join(filtered[-6:]).strip()

    return " ".join(filtered).strip() if filtered else None


def extract_introduction(md_text):
    text = normalize_text(md_text)

    pattern = re.compile(
        r"\b(?:1\.?\s*introduction|introduction)\b(.*?)(?=\b(?:2\.|related work|background|method|methods|approach)\b)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    return clean_section_text(match.group(1).strip()) if match else None


def extract_conclusion(md_text):
    text = normalize_text(md_text)

    pattern = re.compile(
        r"\b(?:conclusion|conclusions)\b(.*?)(?=\b(?:references|acknowledg(?:e)?ments?|appendix|figure\s+\d+|table\s+\d+)\b|$)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return clean_section_text(match.group(1).strip())

    return None


def main():
    records = load_jsonl(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc="Extracting sections", unit="paper"):
            arxiv_id = record["arxiv_id"]
            title = record["title"]
            pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

            if not pdf_path.exists():
                tqdm.write(f"Skipping {arxiv_id}: PDF not found")
                skipped_count += 1
                continue

            try:
                md_text = pymupdf4llm.to_markdown(str(pdf_path))

                abstract = extract_abstract(md_text)
                introduction = extract_introduction(md_text)
                conclusion = extract_conclusion(md_text)

                output_record = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "introduction": introduction,
                    "conclusion": conclusion
                }

                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                saved_count += 1
                tqdm.write(f"Saved {arxiv_id}")

            except Exception as e:
                tqdm.write(f"Error processing {arxiv_id}: {e}")
                skipped_count += 1

    print(f"\nDone. Saved: {saved_count}, Skipped: {skipped_count}")
    print(f"Dataset written to: {OUTPUT_PATH}")