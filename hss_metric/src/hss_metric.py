"""
Hungarian Summary Similarity Metric (HSSM)

This module implements the sentence-level semantic similarity
metric using Hungarian matching.
"""

import nltk
from sentence_transformers import SentenceTransformer


def split_into_sentences(text: str) -> list[str]:
    """
    Split input text into sentences using NLTK's sentence tokenizer.
    """
    return nltk.sent_tokenize(text)


def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence embedding model used by HSSM.
    """
    model = SentenceTransformer("all-mpnet-base-v2")
    return model


def embed_sentences(sentences: list[str], model: SentenceTransformer):
    """
    Convert a list of sentences into embedding vectors.
    """
    embeddings = model.encode(sentences)
    return embeddings


def main():
    sample_text = (
        "Large language models are increasingly used in summarization. "
        "They can generate fluent outputs. "
        "However, evaluation remains difficult. "
        "Existing metrics do not always capture semantic structure."
    )

    sentences = split_into_sentences(sample_text)
    model = load_embedding_model()
    embeddings = embed_sentences(sentences, model)

    print("Sentence count:", len(sentences))
    print("Embedding count:", len(embeddings))
    print("First embedding length:", len(embeddings[0]))

    for i, sentence in enumerate(sentences, start=1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    main()