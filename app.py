import streamlit as st
import pandas as pd
import os
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ================================================
# Career Compass Pakistan
# A beginner-friendly Standard RAG Streamlit MVP
# ================================================

APP_TITLE = "🎓 Career Compass Pakistan"
APP_SUBTITLE = "Your AI-powered guide from education to career"
DATA_DIR = Path(__file__).parent / "data"
TOP_K = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Career Compass Pakistan",
    page_icon="🎓",
    layout="wide"
)

st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# -----------------------
# Dataset check
# -----------------------
files = {
    "Colleges": "colleges.csv",
    "Universities": "universities.csv",
    "Scholarships": "scholarships.csv",
    "Careers": "careers.csv",
}

st.subheader("📊 Dataset Status")

datasets = {}
for name, filename in files.items():
    file_path = DATA_DIR / filename
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                st.success(f"✅ {name}: {len(df)} rows loaded")
                st.write(f"Columns: {len(df.columns)}")
                datasets[name] = df
            else:
                st.warning(f"⚠️ {name}: file found but empty")
        except Exception as e:
            st.error(f"❌ {name} exists but could not be read.")
            st.code(str(e))
    else:
        st.error(f"❌ {name}: {filename} was NOT found")

st.write("Files in data folder:", os.listdir(DATA_DIR) if DATA_DIR.exists() else "No data folder found")

# -----------------------
# Student Assessment
# -----------------------
st.subheader("👤 Student Assessment")

name = st.text_input("Your Name")
interest = st.selectbox("Preferred Field", ["Engineering", "Medicine", "Business", "Arts"])
strength = st.radio("Biggest Strength", ["Math", "Science", "Creativity", "Communication"])

if st.button("Submit Assessment"):
    st.success(f"Thanks {name}! You selected {interest} with strength in {strength}.")

# -----------------------
# Groq Test
# -----------------------
st.subheader("🤖 Groq Test")

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if api_key:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        messages=[{"role": "user", "content": "Hello Groq, test message!"}]
    )
    st.write("Groq says:", response.choices[0].message.content)
else:
    st.error("❌ No Groq API key found. Please set it in Streamlit secrets.")

# -----------------------
# FAISS Setup (placeholder)
# -----------------------
st.subheader("📂 FAISS Index Setup")

try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    st.success("✅ SentenceTransformer loaded")

    # Example: build FAISS index for careers dataset if available
    if "Careers" in datasets:
        texts = datasets["Careers"]["CareerTitle"].astype(str).tolist() if "CareerTitle" in datasets["Careers"].columns else datasets["Careers"].iloc[:,0].astype(str).tolist()
        embeddings = model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        st.success(f"✅ FAISS index built for Careers ({len(texts)} entries)")
    else:
        st.info("ℹ️ Careers dataset not available for FAISS demo")

except Exception as e:
    st.error("❌ Error setting up FAISS")
    st.code(str(e))
