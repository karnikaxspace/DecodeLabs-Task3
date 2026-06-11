
# 🎯 Project 3: AI Recommendation Logic – Tech Stack Recommender

**Author:** Karnika Kumari  
**Batch:** 2026 | DecodeLabs Industrial Training Kit  


---

## 📌 Overview

A **content‑based recommendation engine** that maps a user’s technical skills to the most relevant job roles. Uses **TF‑IDF vectorisation** and **cosine similarity** – no collaborative filtering, no other users' data. Pure pattern alignment.

---

## 🎯 Learning Objectives

- Build a recommendation system using **content‑based filtering**
- Apply **TF‑IDF** to transform text skills into weighted vectors
- Measure similarity using **cosine similarity** (angle between vectors)
- Implement the **IPO model** (Input → Process → Output)
- Bypass the **cold start problem** via user surveys
- Deploy with a **Streamlit UI** and CLI version

---

## 🧮 Mathematical Foundation

### TF‑IDF (Term Frequency – Inverse Document Frequency)

- **TF** – How often a skill appears in a job role (local importance)
- **IDF** – Penalises skills that appear in many job roles (global rarity)

### Cosine Similarity

\[
\text{similarity} = \frac{A \cdot B}{\|A\| \|B\|}
\]

- Measures the **angle** between two vectors – invariant to magnitude.
- Returns values between 0 (unrelated) and 1 (identical).

---

## 📁 Project Structure
    - app.py
    - recommender.py
    - raw_skills.csv
    - requirements.txt
    - README.md

---

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.7+

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2 Run the Web UI (recommended)
``` bash
streamlit run app.py
Enter your skills (comma separated) → click Recommend Jobs → see top matches with similarity scores.
```

### 3. Run the CLI version
``` bash
python recommender.py
```

### 📊 Dataset – raw_skills.csv
| Job Role               | Skills                                                                 |
|------------------------|------------------------------------------------------------------------|
| Cloud Architect        | AWS, Cloud Computing, Automation, DevOps, Python, Infrastructure       |
| Data Scientist         | Python, Statistics, Machine Learning, SQL, Data Visualization, R       |
| Frontend Developer     | JavaScript, React, CSS, HTML, Web Design, UI/UX                        |
| Backend Developer      | Python, Java, Node.js, APIs, Databases, Microservices                  |
| DevOps Engineer        | Docker, Jenkins, Kubernetes, Automation, CI/CD, Cloud, Linux           |
| AI Engineer            | Python, TensorFlow, Machine Learning, Deep Learning, NLP, Computer Vision |
| Cybersecurity Analyst  | Network Security, Risk Analysis, Python, SIEM, Cryptography, Incident Response |
| Full Stack Developer   | JavaScript, Python, React, Node.js, MongoDB, HTML/CSS, REST APIs       |

---

### 🧪 Example Run
User input: Python, Cloud Computing, Automation, Docker

Top 3 recommendations:

Cloud Architect – Similarity: 0.87

DevOps Engineer – Similarity: 0.76

AI Engineer – Similarity: 0.58

---
### ❄️ Cold Start Bypass
User cold start – Avoided by forcing the user to input skills (onboarding survey).

Item cold start – Content‑based filtering naturally handles new items via their metadata.

---
### ✅ Completion Badge
Earned: Content‑Based Filtering | TF‑IDF | Cosine Similarity | Recommendation Pipeline
