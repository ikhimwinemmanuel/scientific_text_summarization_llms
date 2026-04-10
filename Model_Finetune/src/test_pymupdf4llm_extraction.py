from pathlib import Path
import re
import pymupdf4llm

PDF_DIR = Path("Model_Finetune/data/raw/pdfs")


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_abstract(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\babstract\b[:\s]*(.*?)(?=\b(?:1\.?\s*introduction|introduction)\b)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_introduction(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\b(?:1\.?\s*introduction|introduction)\b(.*?)(?=\b(?:2\.|related work|background|method|methods|approach)\b)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_conclusion(text):
    text = normalize_text(text)

    pattern = re.compile(
        r"\b(?:conclusion|conclusions)\b(.*?)(?=\b(?:references|acknowledg(?:e)?ments?|appendix)\b|$)",
        re.IGNORECASE
    )

    match = pattern.search(text)
    return match.group(1).strip() if match else None


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    sample_pdf = pdf_files[0]
    print(f"Testing PDF: {sample_pdf.name}")

    md_text = pymupdf4llm.to_markdown(str(sample_pdf))

    print("\n--- FIRST 2000 CHARACTERS OF MARKDOWN ---\n")
    print(md_text[:2000])

    abstract = extract_abstract(md_text)
    introduction = extract_introduction(md_text)
    conclusion = extract_conclusion(md_text)

    print("\n--- ABSTRACT ---\n")
    print(abstract[:1500] if abstract else "Abstract not found")

    print("\n--- INTRODUCTION ---\n")
    print(introduction[:1500] if introduction else "Introduction not found")

    print("\n--- CONCLUSION ---\n")
    print(conclusion[:1500] if conclusion else "Conclusion not found")


if __name__ == "__main__":
    main()