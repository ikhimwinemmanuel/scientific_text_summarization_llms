import json
import requests
from pathlib import Path

INPUT_PATH = Path("Model_Finetune/data/processed/fixed_papers.jsonl")
PDF_DIR = Path("Model_Finetune/data/raw/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    return records


def download_pdf(url, save_path):
    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            save_path.write_bytes(response.content)
            return True
        else:
            print(f"Failed to download PDF. HTTP status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False


def main():
    records = load_jsonl(INPUT_PATH)

    print(f"Found {len(records)} papers in fixed_papers.jsonl")

    for record in records:
        arxiv_id = record["arxiv_id"]
        pdf_url = record["pdf_url"]
        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

        if pdf_path.exists():
            print(f"Skipping {arxiv_id} (already downloaded)")
            continue

        print(f"Downloading {arxiv_id} ...")
        success = download_pdf(pdf_url, pdf_path)

        if success:
            print(f"Saved to: {pdf_path}")
        else:
            print(f"Download failed for: {arxiv_id}")


if __name__ == "__main__":
    main()