from pathlib import Path
import re
import pymupdf4llm

PDF_DIR = Path("Model_Finetune/data/raw/pdfs")


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def clean_section_text(text):
    if not text:
        return None

    # Remove picture markers and related noise
    text = re.sub(r"\*\*==> picture.*?<==\*\*", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # To remove figure/table captions from the captured section
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

    # to remove markdown symbols that may remain
    text = re.sub(r"[*`#<>_]", " ", text)

    return normalize_text(text)


def extract_abstract(md_text):
    text = normalize_text(md_text)

    # if explicit Abstract heading exists
    explicit_pattern = re.compile(
        r"\babstract\b[:\s]*(.*?)(?=\b(?:1\.?\s*introduction|introduction)\b)",
        re.IGNORECASE
    )
    match = explicit_pattern.search(text)
    if match:
        return clean_section_text(match.group(1).strip())

    # if no explicit Abstract heading
    intro_match = re.search(r"\b(?:1\.?\s*introduction|introduction)\b", text, re.IGNORECASE)
    if not intro_match:
        return None

    before_intro = text[:intro_match.start()].strip()

    # Remove common metadata-like parts
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

    # Keep the later sentence block, which is more likely to be the abstract
    sentences = re.split(r"(?<=[.!?])\s+", candidate)
    if len(sentences) >= 4:
        return " ".join(sentences[-8:]).strip()

    return candidate if candidate else None


def extract_introduction(text):
    text = normalize_text(text)

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