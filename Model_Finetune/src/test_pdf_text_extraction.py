from pathlib import Path
import fitz  # PyMuPDF

PDF_DIR = Path("../data/raw/pdfs")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        full_text.append(page.get_text())

    doc.close()
    return "\n".join(full_text)


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw/pdfs")
        return

    sample_pdf = pdf_files[0]
    print(f"Testing PDF: {sample_pdf.name}")

    text = extract_text_from_pdf(sample_pdf)

    print("\n--- FIRST 2000 CHARACTERS ---\n")
    print(text[:2000])

    print("\n--- TOTAL CHARACTERS ---\n")
    print(len(text))


if __name__ == "__main__":
    main()