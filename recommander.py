"""
Tech Stack Recommender – AI Project 3 (DecodeLabs)

Content‑based filtering using TF‑IDF and cosine similarity.
Maps user skills to the most relevant job roles.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# ---------- Configuration ----------
TOP_N = 3                     # Number of recommendations to show
CSV_PATH = "raw_skills.csv"   # Dataset file


def preprocess_skill_string(skill_phrase: str) -> str:
    """
    Convert a skill phrase into a clean token:
      - lowercased
      - spaces replaced with underscores (to keep multi‑word skills as one token)
    Example: "Cloud Computing" -> "cloud_computing"
    """
    return skill_phrase.strip().lower().replace(" ", "_")


def load_and_prepare_data(csv_path: str):
    """
    Load the CSV, extract job roles and their skill sets.
    Returns:
        job_titles: list of job role names
        corpus: list of space‑separated skill tokens (one string per job role)
    """
    df = pd.read_csv(csv_path)
    job_titles = df["job_role"].tolist()
    corpus = []
    for skills_raw in df["skills"]:
        # Split by comma, clean each skill, and then join with spaces
        skill_list = [preprocess_skill_string(s) for s in skills_raw.split(",")]
        corpus.append(" ".join(skill_list))
    return job_titles, corpus


def get_user_skills() -> list:
    """Prompt user for skills and return a list of cleaned skill strings."""
    print("\n🎯 Tech Stack Recommender – enter your technical skills")
    print("   (e.g., Python, Cloud Computing, Automation)\n")
    raw_input = input("Your skills (comma separated): ").strip()
    if not raw_input:
        print("❌ No skills entered. Please run the script again.")
        sys.exit(1)
    # Split by comma, trim whitespace, and preprocess each skill
    skills = [preprocess_skill_string(s) for s in raw_input.split(",")]
    return skills


def main():
    # ----- Step 1: Ingestion (data & user) -----
    try:
        job_titles, corpus = load_and_prepare_data(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {CSV_PATH}")
        print("Ensure 'raw_skills.csv' is in the same directory as this script.")
        sys.exit(1)

    user_skills = get_user_skills()
    user_skill_string = " ".join(user_skills)

    # ----- Step 2: Vectorisation (TF‑IDF) -----
    # Fit the vectorizer on all job role skill strings
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")   # underscores are kept as part of token
    job_vectors = vectorizer.fit_transform(corpus)

    # Transform user skill string into the same vector space
    user_vector = vectorizer.transform([user_skill_string])

    # ----- Step 3: Scoring (Cosine Similarity) -----
    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    # ----- Step 4: Sorting & Filtering (Top‑N) -----
    # Get indices of top N scores (descending order)
    top_indices = np.argsort(similarity_scores)[::-1][:TOP_N]

    print("\n" + "=" * 50)
    print(f"✨ Top {TOP_N} recommended job roles for your skills:")
    print("=" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        score = similarity_scores[idx]
        # Handle very low scores gracefully
        if score < 0.01:
            print(f"{rank}. {job_titles[idx]} – (similarity too low, try other skills)")
        else:
            print(f"{rank}. {job_titles[idx]:<22} (Similarity: {score:.2f})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()