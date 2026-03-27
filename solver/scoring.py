import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from solver.candidates import cohesion


def margin(candidate, all_candidates):
    """
    Gap between candidate cohesion and best overlapping competitor.
    Higher margin means the group is more unambiguous.
    """
    c_set = candidate["word_set"]
    c_coh = candidate["cohesion"]
    
    best_overlap_coh = 0.0
    for other in all_candidates:
        if other["word_set"] == c_set:
            continue
        if len(c_set & other["word_set"]) >= 1:
            best_overlap_coh = max(best_overlap_coh, other["cohesion"])
    
    return c_coh - best_overlap_coh


def false_group_risk(candidate, all_candidates, tau, delta=0.20):
    """
    Risk penalty based on how many strong false groups overlap with candidate.
    Higher risk means the candidate looks like a known decoy pattern.
    """
    c_set = candidate["word_set"]
    risk = 0
    threshold = tau + delta
    
    for other in all_candidates:
        if other["word_set"] == c_set:
            continue
        if other["cohesion"] >= threshold:
            overlap = len(c_set & other["word_set"])
            if overlap >= 2:
                risk += 1
    
    return risk


def composite_score(candidate, all_candidates, tau,
                     alpha=1.0, beta=0.5, gamma=0.1, delta=0.20):
    """
    Composite scoring function:
    Score(S) = alpha * cohesion + beta * margin - gamma * risk
    
    Args:
        candidate: dict with word_set and cohesion
        all_candidates: full ranked candidate list
        tau: cohesion of weakest true group (estimated as min cohesion
             of top 4 non-overlapping groups during solving)
        alpha, beta, gamma: scoring weights
        delta: false group margin threshold
    
    Returns:
        float score
    """
    coh = candidate["cohesion"]
    mar = margin(candidate, all_candidates)
    risk = false_group_risk(candidate, all_candidates, tau, delta)
    
    return alpha * coh + beta * mar - gamma * risk


def score_all_candidates(candidates, tau, alpha=1.0, beta=0.5, gamma=0.1, delta=0.20):
    """Add composite scores to all candidates and re-sort."""
    for c in candidates:
        c["score"] = composite_score(c, candidates, tau, alpha, beta, gamma, delta)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates