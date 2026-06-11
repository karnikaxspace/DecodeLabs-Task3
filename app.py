"""
Tech Stack Recommender – Web UI (Streamlit)
Project 3: AI Recommendation Logic | DecodeLabs
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Configuration ----------
TOP_N = 5
CSV_PATH = "raw_skills.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

def preprocess_skill_string(skill):
    return skill.strip().lower().replace(" ", "_")

def get_corpus(df):
    corpus = []
    for skills_raw in df["skills"]:
        skill_list = [preprocess_skill_string(s) for s in skills_raw.split(",")]
        corpus.append(" ".join(skill_list))
    return corpus

# ---------- UI ----------
st.set_page_config(page_title="Tech Stack Recommender", page_icon="🎯", layout="centered")

st.title("🎯 Tech Stack Recommender")
st.markdown("### *Content‑Based AI Recommendation Engine*")
st.markdown("Tell me your skills – I’ll find the best job roles for you.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    - **Input**: Your technical skills (e.g., *Python, Cloud Computing, Automation*)
    - **Process**: TF‑IDF vectorisation + Cosine similarity
    - **Output**: Top job roles ranked by mathematical similarity
    - **No other users’ data** – purely content‑based filtering
    """)

# Sidebar
st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Number of recommendations", 1, 10, TOP_N)
st.sidebar.markdown("---")
st.sidebar.caption("Built for DecodeLabs Project 3")

# Main input
skills_input = st.text_input(
    "✏️ Enter your skills (comma separated)",
    placeholder="e.g., Python, Cloud Computing, Automation, Docker"
)

if st.button("🔍 Recommend Jobs", type="primary"):
    if not skills_input.strip():
        st.warning("Please enter at least one skill.")
    else:
        # Process input
        user_skills = [preprocess_skill_string(s) for s in skills_input.split(",")]
        user_skill_string = " ".join(user_skills)
        
        # Load data
        df = load_data()
        job_titles = df["job_role"].tolist()
        corpus = get_corpus(df)
        
        # TF-IDF vectorisation
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        job_vectors = vectorizer.fit_transform(corpus)
        user_vector = vectorizer.transform([user_skill_string])
        
        # Cosine similarity
        scores = cosine_similarity(user_vector, job_vectors).flatten()
        
        # Sort and get top N
        sorted_indices = np.argsort(scores)[::-1][:top_n]
        
        # Display results
        st.subheader("📌 Recommended Job Roles")
        
        cols = st.columns(1)
        with cols[0]:
            for rank, idx in enumerate(sorted_indices, 1):
                score = scores[idx]
                job = job_titles[idx]
                # Progress bar for visual similarity
                st.markdown(f"**{rank}. {job}**")
                st.progress(float(score))
                st.caption(f"Similarity score: `{score:.3f}`")
                st.markdown("---")
        
        # Optional: show all scores (expandable)
        with st.expander("📊 Full similarity scores"):
            results_df = pd.DataFrame({
                "Job Role": job_titles,
                "Similarity": scores
            }).sort_values("Similarity", ascending=False)
            st.dataframe(results_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🧠 Powered by TF‑IDF + Cosine Similarity | Cold‑start bypassed via user survey")