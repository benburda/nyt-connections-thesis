def apply_feedback(candidates, guess, feedback, remaining_words):
    """
    Update candidate list based on feedback from a guess.
    """
    guess_set = frozenset(guess)

    if feedback == "correct":
        # remove all candidates containing any word from correct group
        candidates = [
            c for c in candidates
            if len(c["word_set"] & guess_set) == 0
        ]

    elif feedback == "one_away":
        # remove exact guess, boost candidates sharing 3 words
        candidates = [
            c for c in candidates
            if c["word_set"] != guess_set
        ]
        for c in candidates:
            overlap = len(c["word_set"] & guess_set)
            if overlap == 3:
                c["score"] = c.get("score", c["cohesion"]) * 1.5
        candidates.sort(key=lambda x: x.get("score", x["cohesion"]), reverse=True)

    elif feedback == "incorrect":
        # just remove the exact guess
        candidates = [
            c for c in candidates
            if c["word_set"] != guess_set
        ]

    return candidates


def simulate_feedback(guess, true_groups):
    """
    Simulate NYT feedback for a guess given the true solution.
    
    Args:
        guess: list of 4 words
        true_groups: list of dicts with "members" key
    
    Returns:
        "correct", "one_away", or "incorrect"
    """
    guess_set = frozenset(guess)
    true_sets = [frozenset(g["members"]) for g in true_groups]

    # check if correct
    if guess_set in true_sets:
        return "correct"

    # check if one away
    for true_set in true_sets:
        if len(guess_set & true_set) == 3:
            return "one_away"

    return "incorrect"