import arxiv

QUERY = "cat:cs.*"
MAX_RESULTS = 3


def main():
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    for i, paper in enumerate(client.results(search), start=1):
        arxiv_id = paper.entry_id.split("/")[-1]
        print(f"Paper {i}")
        print(f"arXiv ID: {arxiv_id}")
        print(f"Title: {paper.title}")
        print(f"Published: {paper.published}")
        print("-" * 50)


if __name__ == "__main__":
    main()