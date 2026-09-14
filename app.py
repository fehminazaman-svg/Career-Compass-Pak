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
DATA_DIR = Path(__file__).parent / "data"

files = {
    "Colleges": "colleges.csv",
    "Universities": "universities.csv",
    "Scholarships": "scholarships.csv",
    "Careers": "careers.csv",
}

st.subheader("📊 Dataset Status")

for name, filename in files.items():
    file_path = DATA_DIR / filename

    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                st.success(f"✅ {name}: {len(df)} rows loaded")
                st.write(f"Columns: {len(df.columns)}")
            else:
                st.warning(f"⚠️ {name}: file found but empty")
        except Exception as e:
            st.error(f"❌ {name} exists but could not be read.")
            st.code(str(e))
    else:
        st.error(f"❌ {name}: {filename} was NOT found")

# -----------------------
# Debugging helper
# -----------------------
st.write("Files in data folder:", os.listdir(DATA_DIR) if DATA_DIR.exists() else "No data folder found")
