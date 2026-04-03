import arxiv
import requests
from pathlib import Path

QUERY = "cat:cs.*"
MAX_RESULTS = 3

PDF_DIR = Path("data/raw/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


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
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    for i, paper in enumerate(client.results(search), start=1):
        arxiv_id = paper.entry_id.split("/")[-1]
        pdf_url = paper.pdf_url
        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

        print(f"\nPaper {i}")
        print(f"arXiv ID: {arxiv_id}")
        print(f"Downloading PDF...")

        success = download_pdf(pdf_url, pdf_path)

        if success:
            print(f"Saved to: {pdf_path}")
        else:
            print("Download failed.")


if __name__ == "__main__":
    main()