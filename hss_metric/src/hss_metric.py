"""
Hungarian Summary Similarity Metric (HSSM)

This module implements the sentence-level semantic similarity
metric using Hungarian matching.
"""

import nltk
nltk.download('punkt_tab')

def split_into_sentences(text: str) -> list[str]:
    """
    Split input text into sentences using NLTK's sentence tokenizer.
    """
    return nltk.sent_tokenize(text)


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