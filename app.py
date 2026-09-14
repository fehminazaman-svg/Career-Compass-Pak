import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Career Compass Pakistan",
    page_icon="🎓"
)

st.title("🎓 Career Compass Pakistan")
st.write("Dataset connection test")

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

            st.success(
                f"✅ {name}: {len(df)} rows loaded"
            )

            st.write(
                f"Columns: {len(df.columns)}"
            )

        except Exception as e:

            st.error(
                f"❌ {name} exists but could not be read."
            )

            st.code(str(e))

    else:

        st.error(
            f"❌ {name}: {filename} was NOT found"
        )
