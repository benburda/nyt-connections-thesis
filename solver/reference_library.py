import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def build_reference_library(train_puzzles, model):
    """
    Build a reference library of concept embeddings from training puzzles.
    For each group in the training set, embed its description as a concept vector.
    
    Args:
        train_puzzles: list of training puzzle dicts with groups
        model: SentenceTransformer model
    
    Returns:
        dict with concept embeddings and metadata
    """
    descriptions = []
    metadata = []
    
    for puzzle in train_puzzles:
        for group in puzzle["groups"]:
            description = group["group"]
            members = group["members"]
            level = group["level"]
            
            descriptions.append(description)
            metadata.append({
                "puzzle_id": puzzle["puzzle_id"],
                "description": description,
                "members": members,
                "level": level
            })
    
    print(f"Embedding {len(descriptions)} concept descriptions...")
    concept_embeddings = model.encode(descriptions, show_progress_bar=False)
    
    return {
        "embeddings": concept_embeddings,
        "metadata": metadata,
        "descriptions": descriptions
    }


def reference_score(candidate_words, word_to_emb, library, top_k=5):
    """
    Score a candidate group by similarity to the reference library.
    
    Computes the mean embedding of the candidate group and finds
    the similarity to the closest concept vectors in the library.
    
    Args:
        candidate_words: list of 4 words in the candidate group
        word_to_emb: dict mapping word to embedding
        library: reference library from build_reference_library
        top_k: number of top library matches to average
    
    Returns:
        float similarity score in [0, 1]
    """
    # mean embedding of candidate group
    candidate_embs = np.array([word_to_emb[w] for w in candidate_words])
    candidate_mean = candidate_embs.mean(axis=0, keepdims=True)
    
    # similarity to all concept embeddings
    sims = cosine_similarity(candidate_mean, library["embeddings"])[0]
    
    # average of top_k most similar concepts
    top_sims = np.sort(sims)[-top_k:]
    return float(top_sims.mean())


def add_reference_scores(candidates, word_to_emb, library, top_k=5):
    """Add reference library scores to all candidates."""
    for c in candidates:
        c["reference"] = reference_score(c["words"], word_to_emb, library, top_k)
    return candidates