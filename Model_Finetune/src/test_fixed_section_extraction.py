import json
import re
from pathlib import Path
import fitz  # PyMuPDF

INPUT_PATH = Path("model_finetune/data/processed/fixed_papers.jsonl")
PDF_DIR = Path("model_finetune/data/raw/pdfs")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        full_text.append(page.get_text())

    doc.close()
    return "\n".join(full_text)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_abstract(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\babstract\b[:\s]*(.*?)(?=\b(?:1\.?\s*introduction|introduction)\b)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return None


def extract_introduction(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\b(?:1\.?\s*introduction|introduction)\b(.*?)(?=\b(?:2\.|related work|background|method|methods|approach)\b)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return None


def extract_conclusion(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\b(?:conclusion|conclusions)\b(.*?)(?=\b(?:references|acknowledg(?:e)?ments?|appendix)\b|$)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return None


def main():
    records = load_jsonl(INPUT_PATH)

    if not records:
        print("No records found in fixed_papers.jsonl")
        return

    record = records[0]
    arxiv_id = record["arxiv_id"]
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Testing paper: {arxiv_id}")
    text = extract_text_from_pdf(pdf_path)

    abstract = extract_abstract(text)
    introduction = extract_introduction(text)
    conclusion = extract_conclusion(text)

    print("\n--- ABSTRACT ---\n")
    print(abstract[:1500] if abstract else "Abstract not found")

    print("\n--- INTRODUCTION ---\n")
    print(introduction[:1500] if introduction else "Introduction not found")

    print("\n--- CONCLUSION ---\n")
    print(conclusion[:1500] if conclusion else "Conclusion not found")


if __name__ == "__main__":
    main()