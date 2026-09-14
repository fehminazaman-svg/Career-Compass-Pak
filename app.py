import streamlit as st
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import os

# ================================================
# Career Compass Pakistan
# AI-powered education + career guidance
# ================================================

APP_TITLE = "🎓 Career Compass Pakistan"
APP_SUBTITLE = "Your AI-powered guide from education to career"
DATA_DIR = Path(__file__).parent / "data"
TOP_K = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-70b-8192"   # ✅ Correct model name

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
st.title(APP_TITLE)
st.write(APP_SUBTITLE)
st.info("Career Compass Pakistan provides guidance based on your input and available datasets. Always verify admission, fee and scholarship details from official sources.")

# -----------------------
# Load data with caching
# -----------------------
@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")
            return None
    else:
        return None

datasets = {
    "Colleges": load_csv("colleges.csv"),
    "Universities": load_csv("universities.csv"),
    "Scholarships": load_csv("scholarships.csv"),
    "Careers": load_csv("careers.csv"),
}

st.subheader("📊 Dataset Status")
for name, df in datasets.items():
    if df is None:
        st.error(f"❌ {name}: missing or unreadable")
    elif len(df) == 0:
        st.warning(f"⚠️ {name}: file found but empty")
    else:
        st.success(f"✅ {name}: {len(df)} rows loaded, {len(df.columns)} columns")

# -----------------------
# Embedding model + FAISS
# -----------------------
@st.cache_resource
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

model = get_model()

def build_index(df, label):
    if df is None or len(df) == 0:
        return None, []
    texts = df.astype(str).apply(lambda row: " ".join([str(x) for x in row.values]), axis=1).tolist()
    embeddings = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

indexes = {}
for name, df in datasets.items():
    idx, docs = build_index(df, name)
    indexes[name] = (idx, docs)

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("📌 Navigate")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "👤 Student Assessment",
    "🎯 Recommendations",
    "🎓 Education Explorer",
    "💰 Scholarships & Expenses",
    "💼 Career Opportunities",
    "🗺 My Roadmap"
])

# -----------------------
# Pages
# -----------------------
if page == "🏠 Home":
    st.header("Welcome to Career Compass Pakistan")
    st.write("A simple AI + RAG platform that connects your profile to education options, scholarships, and career paths.")

elif page == "👤 Student Assessment":
    st.header("👤 Student Assessment")
    stage = st.selectbox("Current Education Stage", ["Matric", "Intermediate / College"])
    marks = st.number_input("Marks / Percentage", min_value=0.0, max_value=100.0, step=0.1)
    subjects = st.text_area("Subjects studied")
    interests = st.text_area("Your interests")
    goals = st.text_area("Career dreams / future goals")
    if st.button("Submit Profile"):
        st.session_state["profile"] = {
            "stage": stage,
            "marks": marks,
            "subjects": subjects,
            "interests": interests,
            "goals": goals,
        }
        st.success("Profile saved!")

elif page == "🎯 Recommendations":
    st.header("🎯 AI Recommendations")
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if api_key and "profile" in st.session_state:
        client = Groq(api_key=api_key)
        profile = st.session_state["profile"]
        prompt = f"Student profile: {profile}. Recommend suitable study paths or degree fields."
        try:
            response = client.chat.completions.create(
                model=DEFAULT_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are Career Compass Pakistan, an educational guidance assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq API error: {e}")
    else:
        st.warning("No profile or API key found.")

elif page == "🎓 Education Explorer":
    st.header("🎓 Explore Institutions")
    if indexes["Universities"][0]:
        query = st.text_input("Search for a program or field")
        if query:
            q_emb = model.encode([query], normalize_embeddings=True)
            D, I = indexes["Universities"][0].search(np.array(q_emb), TOP_K)
            for i in I[0]:
                st.write(indexes["Universities"][1][i])

elif page == "💰 Scholarships & Expenses":
    st.header("💰 Scholarships & Expenses")
    if indexes["Scholarships"][0]:
        query = st.text_input("Search scholarships")
        if query:
            q_emb = model.encode([query], normalize_embeddings=True)
            D, I = indexes["Scholarships"][0].search(np.array(q_emb), TOP_K)
            for i in I[0]:
                st.write(indexes["Scholarships"][1][i])

elif page == "💼 Career Opportunities":
    st.header("💼 Career Opportunities")
    if indexes["Careers"][0]:
        query = st.text_input("Search careers")
        if query:
            q_emb = model.encode([query], normalize_embeddings=True)
            D, I = indexes["Careers"][0].search(np.array(q_emb), TOP_K)
            for i in I[0]:
                st.write(indexes["Careers"][1][i])

elif page == "🗺 My Roadmap":
    st.header("🗺 Personalized Roadmap")
    if "profile" in st.session_state:
        st.write("Your Current Position → Recommended Path → Institutions → Scholarships → Careers → Roadmap")
    else:
        st.info("Complete the Student Assessment first.")

