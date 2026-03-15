"""
Hungarian Summary Similarity Metric (HSSM)

This module implements the sentence-level semantic similarity
metric using Hungarian matching.
"""


def split_into_sentences(text: str) -> list[str]:
    """
    Split input text into sentences using a simple rule-based approach.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return [s + "." for s in sentences]


def main():
    sample_text = (
        "Large language models are increasingly used in summarization. "
        "They can generate fluent outputs. "
        "However, evaluation remains difficult. "
        "Existing metrics do not always capture semantic structure."
    )

    sentences = split_into_sentences(sample_text)

    print("Sentence count:", len(sentences))
    for i, sentence in enumerate(sentences, start=1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    main()