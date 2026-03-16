"""
Hungarian Summary Similarity Metric (HSSM)

This module implements the sentence-level semantic similarity
metric using Hungarian matching.
"""

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def build_similarity_matrix(ref_sentences: list[str], gen_sentences: list[str], model: SentenceTransformer):
    """
    Build a sentence-level cosine similarity matrix between
    reference and generated summary sentences.
    """
    ref_embeddings = embed_sentences(ref_sentences, model)
    gen_embeddings = embed_sentences(gen_sentences, model)

    similarity_matrix = cosine_similarity(ref_embeddings, gen_embeddings)
    return similarity_matrix


def main():
    reference_text = (
        "Large language models are increasingly used in summarization. "
        "They can produce fluent summaries. "
        "However, evaluating summary quality remains difficult. "
        "Existing metrics do not always capture semantic meaning."
    )

    generated_text = (
        "Large language models are widely applied in summarization tasks. "
        "They often generate fluent outputs. "
        "Still, summary evaluation remains challenging. "
        "Traditional metrics may fail to capture meaning."
    )

    ref_sentences = split_into_sentences(reference_text)
    gen_sentences = split_into_sentences(generated_text)

    model = load_embedding_model()
    similarity_matrix = build_similarity_matrix(ref_sentences, gen_sentences, model)

    print("Reference sentence count:", len(ref_sentences))
    print("Generated sentence count:", len(gen_sentences))
    print("Similarity matrix shape:", similarity_matrix.shape)
    print("\nSimilarity matrix:")
    print(similarity_matrix)


if __name__ == "__main__":
    main()