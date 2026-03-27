import nltk
from nltk.corpus import wordnet as wn

# download wordnet data if not already present
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def get_synsets(word):
    """Return all WordNet synsets for a word."""
    return wn.synsets(word.lower().replace(" ", "_"))


def get_hypernyms(word):
    """Return set of hypernym lemma names for a word."""
    hypernyms = set()
    for syn in get_synsets(word):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                hypernyms.add(lemma.name().lower())
    return hypernyms


def get_lemmas(word):
    """Return set of lemma names across all synsets."""
    lemmas = set()
    for syn in get_synsets(word):
        for lemma in syn.lemmas():
            lemmas.add(lemma.name().lower())
    return lemmas


def lexical_overlap(words):
    """
    Compute lexical cohesion for a group of words using WordNet.
    Measures how much shared hypernym and lemma structure exists.

    Returns a float in [0, 1].
    """
    if len(words) < 2:
        return 0.0

    word_hypernyms = [get_hypernyms(w) for w in words]
    word_lemmas = [get_lemmas(w) for w in words]

    # count pairs with shared hypernyms or lemmas
    n_pairs = 0
    n_shared = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            n_pairs += 1
            shared_hyper = word_hypernyms[i] & word_hypernyms[j]
            shared_lemma = word_lemmas[i] & word_lemmas[j]
            if shared_hyper or shared_lemma:
                n_shared += 1

    return n_shared / n_pairs if n_pairs > 0 else 0.0


def lexical_score(candidate):
    """
    Compute lexical score for a candidate group.
    Returns float in [0, 1].
    """
    return lexical_overlap(candidate["words"])


def add_lexical_scores(candidates):
    """Add lexical scores to all candidates."""
    for c in candidates:
        c["lexical"] = lexical_score(c)
    return candidates