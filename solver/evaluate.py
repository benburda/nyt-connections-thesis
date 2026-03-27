import numpy as np
from solver.candidates import generate_candidates
from solver.search import greedy_solve, beam_solve, get_tau_estimate
from solver.scoring import score_all_candidates
from solver.lexical import add_lexical_scores
from solver.feedback import apply_feedback, simulate_feedback


def solve_puzzle(puzzle, solver="greedy", beam_width=25,
                 alpha=1.0, beta=0.5, gamma=0.1, eta=0.3,
                 delta=0.20, use_lexical=False, use_feedback=False):
    """
    Run a solver on a single puzzle.

    Args:
        puzzle: puzzle dict with words and embeddings
        solver: "greedy" or "beam"
        beam_width: beam width for beam search
        alpha, beta, gamma, eta: scoring weights
        delta: false group threshold
        use_lexical: whether to add WordNet lexical scores
        use_feedback: whether to use feedback-aware mode

    Returns:
        dict with predicted groups and evaluation metrics
    """
    words = puzzle["words"]
    embeddings = np.array(puzzle["embeddings"])
    true_groups = puzzle["groups"]

    # stage 1: generate candidates
    candidates = generate_candidates(words, embeddings)

    # stage 2: score candidates
    tau = get_tau_estimate(candidates)
    candidates = score_all_candidates(candidates, tau, alpha, beta, gamma, delta)

    # stage 3: add lexical scores if requested
    if use_lexical:
        candidates = add_lexical_scores(candidates)
        for c in candidates:
            c["score"] = c["score"] + eta * c["lexical"]
        candidates.sort(key=lambda x: x["score"], reverse=True)

    # stage 4: solve
    if not use_feedback:
        if solver == "greedy":
            predicted = greedy_solve(words, embeddings, candidates)
        else:
            predicted = beam_solve(words, embeddings, candidates, beam_width)
        return evaluate_prediction(predicted, true_groups)

    else:
        # feedback-aware mode
        return solve_with_feedback(words, embeddings, candidates,
                                   true_groups, beam_width)


def solve_with_feedback(words, embeddings, candidates, true_groups, beam_width=25):
    """
    Solve a puzzle using iterative feedback simulation.
    Makes up to 4 guesses, updating candidates after each.
    """
    remaining = set(words)
    solved_groups = []
    guesses = []
    feedbacks = []
    max_guesses = 4 + (4 - 1)  # 4 correct + up to 3 incorrect

    for _ in range(max_guesses):
        if len(solved_groups) == 4:
            break

        # get current best guess from beam search
        active_candidates = [
            c for c in candidates
            if c["word_set"].issubset(remaining)
        ]
        if not active_candidates:
            break

        predicted = beam_solve(list(remaining), embeddings, active_candidates, beam_width)
        guess = predicted[0]
        guesses.append(guess)

        # simulate feedback
        feedback = simulate_feedback(guess, true_groups)
        feedbacks.append(feedback)

        if feedback == "correct":
            solved_groups.append(guess)
            remaining -= set(guess)
            candidates = apply_feedback(candidates, guess, "correct", remaining)
        else:
            candidates = apply_feedback(candidates, guess, feedback, remaining)

    return evaluate_prediction(solved_groups, true_groups, guesses, feedbacks)


def evaluate_prediction(predicted, true_groups, guesses=None, feedbacks=None):
    """
    Compute evaluation metrics for a prediction.

    Metrics:
        - n_correct: number of correctly identified groups
        - solved: whether all 4 groups were correctly identified
        - top1_correct: whether the first guess was correct
    """
    true_sets = [frozenset(g["members"]) for g in true_groups]
    pred_sets = [frozenset(g) for g in predicted]

    n_correct = sum(1 for p in pred_sets if p in true_sets)
    solved = n_correct == 4
    top1_correct = len(pred_sets) > 0 and pred_sets[0] in true_sets

    return {
        "n_correct": n_correct,
        "solved": solved,
        "top1_correct": top1_correct,
        "n_guesses": len(guesses) if guesses else len(predicted),
        "feedbacks": feedbacks or []
    }


def run_evaluation(puzzles, solver="greedy", beam_width=25,
                   alpha=1.0, beta=0.5, gamma=0.1, eta=0.3,
                   delta=0.20, use_lexical=False, use_feedback=False):
    """
    Run a solver on a list of puzzles and return aggregate metrics.
    """
    results = []
    for puzzle in puzzles:
        result = solve_puzzle(
            puzzle, solver=solver, beam_width=beam_width,
            alpha=alpha, beta=beta, gamma=gamma, eta=eta,
            delta=delta, use_lexical=use_lexical, use_feedback=use_feedback
        )
        results.append(result)

    solve_rate = np.mean([r["solved"] for r in results])
    mean_correct = np.mean([r["n_correct"] for r in results])
    top1_rate = np.mean([r["top1_correct"] for r in results])

    return {
        "solve_rate": solve_rate,
        "mean_correct_groups": mean_correct,
        "top1_accuracy": top1_rate,
        "n_puzzles": len(results),
        "results": results
    }