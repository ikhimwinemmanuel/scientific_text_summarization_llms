import arxiv
import json
from pathlib import Path

QUERY = "cat:cs.*"
MAX_RESULTS = 10

OUTPUT_PATH = Path("Model_Finetune/data/processed/fixed_papers.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, paper in enumerate(client.results(search), start=1):
            record = {
                "no": i,
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "pdf_url": paper.pdf_url,
                "published": str(paper.published) if paper.published else None
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Saved paper {i}: {record['arxiv_id']}")

    print(f"\nFixed paper list saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()