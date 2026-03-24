"""
Hungarian Summary Similarity Metric (HSSM)

This module implements the sentence-level semantic similarity
metric using Hungarian matching.
"""

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np


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
    similarity_matrix = np.maximum(similarity_matrix, 0)

    return similarity_matrix


def apply_hungarian_matching(similarity_matrix):
    """
    Apply Hungarian matching to maximize total sentence similarity.
    """
    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


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
    row_ind, col_ind = apply_hungarian_matching(similarity_matrix)

    print("Hungarian matches:")
    for r, c in zip(row_ind, col_ind):
        print(
            f"R{r+1} -> G{c+1} | "
            f"similarity = {similarity_matrix[r, c]:.4f}"
        )


if __name__ == "__main__":
    main()