import numpy as np
from solver.candidates import generate_candidates, cohesion
from solver.scoring import score_all_candidates


def greedy_solve(words, embeddings, scored_candidates=None):
    """
    Solver 1: Greedy baseline.
    Pick the highest scoring non-overlapping groups one at a time.
    No backtracking.
    """
    if scored_candidates is None:
        candidates = generate_candidates(words, embeddings)
    else:
        candidates = scored_candidates

    predicted = []
    used = set()

    while len(predicted) < 4:
        for c in candidates:
            if len(c["word_set"] & used) == 0:
                predicted.append(c["words"])
                used |= c["word_set"]
                break

    return predicted


def beam_solve(words, embeddings, scored_candidates=None, beam_width=25):
    """
    Solver 2/3: Beam search over full partitions (optimized).
    """
    if scored_candidates is None:
        candidates = generate_candidates(words, embeddings)
    else:
        candidates = scored_candidates

    words_set = set(words)

    # filter to only candidates using words in current word set
    valid_candidates = [
        c for c in candidates
        if c["word_set"].issubset(words_set)
    ]

    if not valid_candidates:
        return []

    # pre-sort by score
    sorted_candidates = sorted(
        valid_candidates,
        key=lambda x: x.get("score", x["cohesion"]),
        reverse=True
    )

    beam = [{"groups": [], "used": frozenset(), "score": 0.0}]

    for _ in range(4):
        if not beam:
            break
        next_beam = []
        for state in beam:
            count = 0
            for c in sorted_candidates:
                if len(c["word_set"] & state["used"]) == 0:
                    next_beam.append({
                        "groups": state["groups"] + [c["words"]],
                        "used": state["used"] | c["word_set"],
                        "score": state["score"] + c.get("score", c["cohesion"])
                    })
                    count += 1
                    if count >= 50:
                        break

        if not next_beam:
            break
        next_beam.sort(key=lambda x: x["score"], reverse=True)
        beam = next_beam[:beam_width]

    if not beam or not beam[0]["groups"]:
        return []

    return beam[0]["groups"]


def get_tau_estimate(candidates):
    """
    Estimate tau (weakest true group cohesion) as the minimum cohesion
    among the top 4 non-overlapping groups found greedily.
    """
    used = set()
    top_groups = []
    for c in candidates:
        if len(c["word_set"] & used) == 0:
            top_groups.append(c)
            used |= c["word_set"]
        if len(top_groups) == 4:
            break
    return min(g["cohesion"] for g in top_groups)