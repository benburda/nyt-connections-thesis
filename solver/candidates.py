import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


def cohesion(embeddings):
    """Mean pairwise cosine similarity within a group."""
    sims = cosine_similarity(embeddings)
    n = len(embeddings)
    total = sum(sims[i][j] for i in range(n) for j in range(i+1, n))
    return total / (n * (n - 1) / 2)


def generate_candidates(words, embeddings):
    """
    Enumerate all C(n,4) candidate groups and compute cohesion for each.
    
    Args:
        words: list of 16 words in the puzzle
        embeddings: np.array of shape (16, d)
    
    Returns:
        List of dicts sorted by cohesion descending:
        [{"words": [...], "cohesion": float}, ...]
    """
    word_to_emb = {w: embeddings[i] for i, w in enumerate(words)}
    
    candidates = []
    for combo in combinations(words, 4):
        embs = np.array([word_to_emb[w] for w in combo])
        c = cohesion(embs)
        candidates.append({
            "words": list(combo),
            "word_set": frozenset(combo),
            "cohesion": c
        })
    
    candidates.sort(key=lambda x: x["cohesion"], reverse=True)
    return candidates


def get_embeddings_dict(words, embeddings):
    """Returns a word -> embedding dictionary."""
    return {w: embeddings[i] for i, w in enumerate(words)}