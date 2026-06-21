"""
ranker.py
Embedding + similarity matching engine.

Uses TF-IDF vectorization (term-frequency / inverse-document-frequency)
as the embedding space, and cosine similarity to rank resumes against
a job description. This is a fully offline, dependency-light embedding
approach — each document becomes a high-dimensional weighted vector,
and similarity is measured as the cosine of the angle between vectors,
exactly the same comparison used with neural embeddings, just with a
classical (and fast, transparent, reproducible) vectorizer underneath.
"""

import re
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# A small set of generic resume/JD boilerplate words that add noise
# to similarity scoring without signaling actual fit.
STOPWORDS_EXTRA = {
    "resume", "curriculum", "vitae", "cv", "page", "references",
    "available", "request", "phone", "email", "address",
}


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rank_resumes(job_description: str, resume_texts: List[str]) -> Dict:
    """
    Rank resumes against a job description using TF-IDF embeddings
    and cosine similarity.

    Returns a dict with:
      - scores: list of similarity scores (0-100) aligned to resume_texts
      - keywords: list of top matched keywords per resume
      - jd_top_terms: top weighted terms from the job description
    """
    if not resume_texts:
        return {"scores": [], "keywords": [], "jd_top_terms": []}

    documents = [_preprocess(job_description)] + [_preprocess(t) for t in resume_texts]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Degenerate case: empty vocabulary (e.g. all text filtered out)
        n = len(resume_texts)
        return {"scores": [0.0] * n, "keywords": [[] for _ in range(n)], "jd_top_terms": []}

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    # Raw cosine similarity on TF-IDF vectors is naturally compressed —
    # even an excellent resume-to-JD match typically lands around 0.25-0.40,
    # since two documents are never near-duplicates of each other. A flat
    # linear stretch either crushes good matches or blows out weak ones,
    # so we apply a square-root curve: it lifts the low-to-mid range
    # (where real signal lives) while still preserving rank order and
    # letting genuinely poor matches stay low.
    scores = [round(min((float(s) ** 0.5) * 120, 99.0), 1) for s in similarities]

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Top JD terms — what the algorithm thinks the role is "about"
    jd_row = tfidf_matrix[0].toarray().flatten()
    top_jd_idx = jd_row.argsort()[::-1][:12]
    jd_top_terms = [feature_names[i] for i in top_jd_idx if jd_row[i] > 0]
    jd_top_terms = [t for t in jd_top_terms if t not in STOPWORDS_EXTRA]

    # For each resume, find which of the JD's top terms it actually contains,
    # weighted by that resume's own TF-IDF score for the term — this becomes
    # the "matched keywords" explanation shown to the recruiter.
    jd_term_set = set(jd_top_terms)
    keywords_per_resume = []
    for i in range(resume_vectors.shape[0]):
        row = resume_vectors[i].toarray().flatten()
        nonzero_idx = row.argsort()[::-1]
        matched = []
        for idx in nonzero_idx:
            term = feature_names[idx]
            if row[idx] <= 0:
                break
            if term in jd_term_set and term not in STOPWORDS_EXTRA:
                matched.append(term)
            if len(matched) >= 8:
                break
        keywords_per_resume.append(matched)

    return {
        "scores": scores,
        "keywords": keywords_per_resume,
        "jd_top_terms": jd_top_terms,
    }


def score_band(score: float) -> str:
    """Categorize a 0-100 score into a qualitative band for the UI."""
    if score >= 60:
        return "strong"
    elif score >= 35:
        return "moderate"
    else:
        return "weak"
