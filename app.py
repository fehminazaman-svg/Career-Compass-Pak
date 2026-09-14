import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Career Compass Pakistan
# A beginner-friendly Standard RAG Streamlit MVP
# ============================================================

APP_TITLE = "🎓 Career Compass Pakistan"
APP_SUBTITLE = "Your AI-powered guide from education to career"
DATA_DIR = Path(__file__).parent / "data"
TOP_K = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Career Compass Pakistan",
    page_icon="🎓",
    layout="wide",
)


# -----------------------------
# Session state
# -----------------------------
DEFAULT_STATE = {
    "student_profile": {},
    "ai_recommendations": [],
    "selected_path": None,
    "selected_institution": None,
    "selected_program": None,
    "retrieved_scholarships": [],
    "retrieved_careers": [],
    "roadmap": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    """Load one CSV safely and normalize missing values."""
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    # Convert column names to readable strings without assuming exact names.
    df.columns = [str(c).strip() for c in df.columns]

    # Replace NaN/inf values so they are safe for display and text conversion.
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")
    return df


@st.cache_data(show_spinner=False)
def load_all_data() -> Dict[str, pd.DataFrame]:
    return {
        "colleges": load_csv("data/colleges.csv"),
        "universities": load_csv("data/universities.csv"),
        "scholarships": load_csv("data/scholarships.csv"),
        "careers": load_csv("data/careers.csv"),
    }



def data_status(data: Dict[str, pd.DataFrame]) -> None:
    """Show a compact dataset status in the sidebar."""
    st.sidebar.markdown("### 📁 Dataset status")

    labels = {
        "colleges": "Colleges",
        "universities": "Universities",
        "scholarships": "Scholarships",
        "careers": "Careers",
    }

    for key, label in labels.items():
        df = data[key]
        if df.empty:
            st.sidebar.warning(f"{label}: missing/empty")
        else:
            st.sidebar.success(f"{label}: {len(df):,} records")


# ============================================================
# Flexible column helpers
# ============================================================

def normalize_text(value: Any) -> str:
    """Convert any value into clean searchable text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """
    Find a likely column without requiring exact CSV column names.
    Exact normalized matches are preferred, then substring matches.
    """
    if df.empty:
        return None

    normalized = {
        normalized_column_name(col): col
        for col in df.columns
    }

    alias_norms = [normalized_column_name(a) for a in aliases]

    for alias in alias_norms:
        if alias in normalized:
            return normalized[alias]

    for col_norm, original in normalized.items():
        for alias in alias_norms:
            if alias and (alias in col_norm or col_norm in alias):
                return original

    return None


def get_value(row: pd.Series, aliases: List[str]) -> str:
    col = find_column(pd.DataFrame([row]), aliases)
    return normalize_text(row[col]) if col else ""


def row_to_document(
    row: pd.Series,
    data_type: str,
    row_index: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Convert a CSV row into a meaningful document while retaining metadata.
    All factual content comes directly from the row.
    """
    parts = [f"TYPE: {data_type.upper()}"]

    metadata: Dict[str, Any] = {
        "data_type": data_type,
        "row_index": int(row_index),
        "institution": "",
        "program": "",
        "source": "",
    }

    # Preserve every non-empty CSV field so unknown/custom columns are not lost.
    for col in row.index:
        value = normalize_text(row[col])
        if not value:
            continue
        parts.append(f"{col}: {value}")

    institution = get_value(
        row,
        [
            "institution",
            "university",
            "university name",
            "college",
            "college name",
            "provider",
            "organization",
        ],
    )
    program = get_value(
        row,
        [
            "program",
            "program name",
            "degree",
            "degree/program",
            "degree name",
            "course",
            "stream",
            "field",
            "study field",
        ],
    )
    source = get_value(
        row,
        [
            "source",
            "official source",
            "source url",
            "official website",
            "link",
            "url",
        ],
    )

    metadata["institution"] = institution
    metadata["program"] = program
    metadata["source"] = source

    return "\n".join(parts), metadata


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    text: str,
    max_words: int = 450,
    overlap_words: int = 70,
) -> List[str]:
    """
    Keep short CSV records together.
    Longer records are split into overlapping word chunks.
    """
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    step = max_words - overlap_words

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += max(step, 1)

    return chunks


def build_documents(
    df: pd.DataFrame,
    data_type: str,
) -> List[Dict[str, Any]]:
    """Create chunked documents and preserve metadata."""
    documents: List[Dict[str, Any]] = []

    if df.empty:
        return documents

    for row_index, row in df.iterrows():
        full_text, metadata = row_to_document(row, data_type, int(row_index))
        chunks = chunk_text(full_text)

        for chunk_index, chunk in enumerate(chunks):
            item = dict(metadata)
            item["chunk_index"] = chunk_index
            item["text"] = chunk
            documents.append(item)

    return documents


# ============================================================
# Embeddings + FAISS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def make_embeddings(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype="float32")

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


@st.cache_resource(show_spinner="Building RAG indexes...")
def build_all_indexes(
    colleges_records: Tuple[Tuple[str, str, str, str, str, str], ...],
    universities_records: Tuple[Tuple[str, str, str, str, str, str], ...],
    scholarships_records: Tuple[Tuple[str, str, str, str, str, str], ...],
    careers_records: Tuple[Tuple[str, str, str, str, str, str], ...],
):
    """
    Build four separate FAISS cosine-similarity indexes.
    Tuples are used because Streamlit cache_resource needs hashable inputs.
    """
    model = get_embedding_model()

    datasets = {
        "colleges": colleges_records,
        "universities": universities_records,
        "scholarships": scholarships_records,
        "careers": careers_records,
    }

    indexes = {}
    documents = {}

    for name, records in datasets.items():
        docs = [
            {
                "text": r[0],
                "data_type": r[1],
                "row_index": int(r[2]),
                "institution": r[3],
                "program": r[4],
                "source": r[5],
            }
            for r in records
        ]

        documents[name] = docs

        if not docs:
            indexes[name] = None
            continue

        vectors = make_embeddings([d["text"] for d in docs], model)

        # Normalized vectors + inner product = cosine similarity.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        indexes[name] = index

    return indexes, documents


def dataframe_to_records(
    df: pd.DataFrame,
    data_type: str,
) -> Tuple[Tuple[str, str, str, str, str, str], ...]:
    docs = build_documents(df, data_type)

    return tuple(
        (
            d["text"],
            d["data_type"],
            str(d["row_index"]),
            d["institution"],
            d["program"],
            d["source"],
        )
        for d in docs
    )


def get_indexes_and_documents(
    data: Dict[str, pd.DataFrame],
):
    return build_all_indexes(
        dataframe_to_records(data["colleges"], "college"),
        dataframe_to_records(data["universities"], "university"),
        dataframe_to_records(data["scholarships"], "scholarship"),
        dataframe_to_records(data["careers"], "career"),
    )


# ============================================================
# Retrieval
# ============================================================

def keyword_score(text: str, query: str) -> float:
    """Small deterministic bonus for exact keyword overlap."""
    text_lower = text.lower()
    query_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(term) > 2
    }

    if not query_terms:
        return 0.0

    hits = sum(1 for term in query_terms if term in text_lower)
    return hits / len(query_terms)


def retrieve_chunks(
    query: str,
    domain: str,
    indexes: Dict[str, Any],
    documents: Dict[str, List[Dict[str, Any]]],
    top_k: int = TOP_K,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Standard RAG retrieval:
    query -> embedding -> FAISS similarity search -> small keyword/filter boost.
    """
    index = indexes.get(domain)
    docs = documents.get(domain, [])

    if index is None or not docs or not query.strip():
        return []

    model = get_embedding_model()
    query_vector = make_embeddings([query], model)

    # Search more than top_k so deterministic filtering/boosting has candidates.
    search_k = min(max(top_k * 4, 20), len(docs))
    scores, ids = index.search(query_vector, search_k)

    candidates = []

    for similarity, doc_id in zip(scores[0], ids[0]):
        if doc_id < 0:
            continue

        doc = dict(docs[int(doc_id)])
        text = doc["text"]

        filter_bonus = 0.0
        if filters:
            for field, value in filters.items():
                value = normalize_text(value).lower()
                if not value:
                    continue

                # Search the document text because schemas differ between CSVs.
                if value in text.lower():
                    filter_bonus += 0.12

        lexical_bonus = keyword_score(text, query) * 0.10
        doc["_score"] = float(similarity) + filter_bonus + lexical_bonus
        candidates.append(doc)

    candidates.sort(key=lambda x: x["_score"], reverse=True)

    # Remove exact duplicate chunks.
    seen = set()
    results = []

    for doc in candidates:
        key = (
            doc.get("data_type"),
            doc.get("row_index"),
            doc.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(doc)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# Groq
# ============================================================

def get_groq_client() -> Optional[Groq]:
    """Create the Groq client only when the secret is available."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        return None

    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def get_groq_model() -> str:
    """Allow the model to be overridden through Streamlit secrets."""
    try:
        model = st.secrets.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        return str(model).strip() or DEFAULT_GROQ_MODEL
    except Exception:
        return DEFAULT_GROQ_MODEL


def call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Optional[str]:
    client = get_groq_client()

    if client is None:
        st.warning(
            "Groq is not configured. Add GROQ_API_KEY to Streamlit Community Cloud "
            "Secrets to enable AI recommendations and roadmap generation."
        )
        return None

    try:
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        st.error(f"Groq request failed: {exc}")
        return None


GROUNDED_SYSTEM_PROMPT = """
You are Career Compass Pakistan, an educational and career guidance assistant.

Use the supplied student profile and retrieved context carefully.

FACTUAL GROUNDING RULES:
- The retrieved CSV context is the factual source for institutions, programs,
  scholarships, eligibility, fees, careers, jobs, and salaries.
- Never invent a university, college, scholarship, program, fee, deadline,
  eligibility rule, job fact, or salary.
- If a requested factual detail is absent from the retrieved context, say that
  it is not available in the current dataset.
- Recommendations are AI-guided reasoning, not a psychological or scientific
  test result.
- Clearly distinguish reasoning/recommendations from verified dataset facts.
- Use simple, student-friendly English.
- Do not claim that a student definitely qualifies for a scholarship or admission.
- Encourage verification of current admission, fee, and scholarship information
  from the official source when a source is available.
"""


def parse_json_response(text: Optional[str]) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return None

    cleaned = text.strip()

    # Remove Markdown code fences if the model used them.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON object or array.
    object_match = re.search(r"(\{.*\})", cleaned, flags=re.S)
    array_match = re.search(r"(\[.*\])", cleaned, flags=re.S)

    for match in [array_match, object_match]:
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


# ============================================================
# Student assessment
# ============================================================

def create_student_profile() -> Dict[str, Any]:
    st.subheader("👤 Student Assessment")
    st.write(
        "Tell us about your studies, interests and goals. The assessment is an "
        "AI-guided compatibility estimate, not a scientific test."
    )

    with st.form("student_assessment_form"):
        stage = st.selectbox(
            "Current education stage",
            ["Matric", "Intermediate / College"],
        )

        marks_label = (
            "Current marks / percentage"
            if stage == "Matric"
            else "Current marks / percentage / CGPA"
        )

        marks = st.text_input(
            marks_label,
            placeholder="Example: 82% or 3.4 CGPA",
        )

        subjects = st.text_area(
            "Subjects studied",
            placeholder="Example: Mathematics, Physics, Chemistry, Computer Science",
            height=80,
        )

        favorite_subjects = st.multiselect(
            "Favorite subjects",
            [
                "Mathematics",
                "Computer Science",
                "Physics",
                "Chemistry",
                "Biology",
                "English",
                "Business / Accounting",
                "Economics",
                "Social Sciences",
                "Arts / Humanities",
                "Other",
            ],
        )

        interests = st.text_area(
            "Interests",
            placeholder="What topics or activities do you enjoy?",
            height=80,
        )

        skills = st.text_area(
            "Skills",
            placeholder="Example: coding, communication, drawing, problem solving",
            height=80,
        )

        strengths = st.text_area(
            "Strengths",
            placeholder="Example: numbers, creativity, teamwork, explaining ideas",
            height=80,
        )

        dislikes = st.text_area(
            "Things you dislike",
            placeholder="Optional",
            height=60,
        )

        col1, col2 = st.columns(2)

        with col1:
            preferred_city = st.text_input(
                "Preferred city",
                placeholder="Optional, e.g. Lahore",
            )

            sector_preference = st.multiselect(
                "Institution preference",
                ["Public", "Private"],
            )

        with col2:
            budget = st.text_input(
                "Approximate education budget",
                placeholder="Optional, e.g. Rs 100,000 per semester",
            )

            working_style = st.multiselect(
                "Preferred working style",
                [
                    "Government job",
                    "Private sector",
                    "Remote work",
                    "Freelancing",
                    "Research",
                    "Entrepreneurship",
                    "International career",
                    "High-income career",
                ],
            )

        career_goals = st.text_area(
            "Career goals",
            placeholder="What kind of future do you want?",
            height=70,
        )

        dream_career = st.text_input(
            "Dream career",
            placeholder="Optional",
        )

        anything_else = st.text_area(
            "Tell us anything else about yourself",
            placeholder="Optional",
            height=80,
        )

        submitted = st.form_submit_button(
            "✨ Analyze my profile",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        profile = {
            "education_stage": stage,
            "marks": marks,
            "subjects": subjects,
            "favorite_subjects": favorite_subjects,
            "interests": interests,
            "skills": skills,
            "strengths": strengths,
            "dislikes": dislikes,
            "preferred_city": preferred_city,
            "institution_preference": sector_preference,
            "budget": budget,
            "working_style": working_style,
            "career_goals": career_goals,
            "dream_career": dream_career,
            "anything_else": anything_else,
        }

        st.session_state.student_profile = profile
        st.session_state.ai_recommendations = []
        st.session_state.selected_path = None
        st.session_state.selected_institution = None
        st.session_state.selected_program = None
        st.session_state.retrieved_scholarships = []
        st.session_state.retrieved_careers = []
        st.session_state.roadmap = ""

        return profile

    return st.session_state.student_profile


def profile_to_text(profile: Dict[str, Any]) -> str:
    lines = []

    for key, value in profile.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)

        value = normalize_text(value)
        if value:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


# ============================================================
# AI recommendations
# ============================================================

def get_ai_recommendations(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not profile:
        return []

    stage = profile.get("education_stage", "")

    if stage == "Matric":
        task = """
Recommend the top 3 suitable Intermediate study paths for this Matric student.
Possible directions include FSc Pre-Medical, FSc Pre-Engineering, ICS, ICom,
FA/Humanities, and other relevant directions. Do not force these examples if
the profile points elsewhere.
"""
    else:
        task = """
Recommend the top 3 suitable degree fields/program directions for this
Intermediate student. Consider the student's stream, subjects, marks,
interests, strengths, skills and goals. Examples include computing, AI,
data science, engineering, business, natural sciences, health sciences and
social sciences, but do not limit the answer to those examples.
"""

    system = GROUNDED_SYSTEM_PROMPT + """
For this recommendation step, the student profile is the primary input.
Do not present compatibility as certain. Return JSON only.
"""

    user = f"""
{task}

Student profile:
{profile_to_text(profile)}

Return exactly this JSON structure:
[
  {{
    "name": "recommended path or degree field",
    "fit": "2-4 sentence explanation of why it matches",
    "future_fields": ["field 1", "field 2"],
    "career_directions": ["direction 1", "direction 2"]
  }}
]

Keep it to exactly 3 recommendations where possible.
"""

    raw = call_groq(system, user, temperature=0.35)
    parsed = parse_json_response(raw)

    if not isinstance(parsed, list):
        return []

    cleaned = []

    for item in parsed[:3]:
        if not isinstance(item, dict):
            continue

        name = normalize_text(item.get("name"))
        if not name:
            continue

        future_fields = item.get("future_fields", [])
        career_directions = item.get("career_directions", [])

        if not isinstance(future_fields, list):
            future_fields = [str(future_fields)]

        if not isinstance(career_directions, list):
            career_directions = [str(career_directions)]

        cleaned.append(
            {
                "name": name,
                "fit": normalize_text(item.get("fit")),
                "future_fields": [
                    normalize_text(x) for x in future_fields if normalize_text(x)
                ],
                "career_directions": [
                    normalize_text(x)
                    for x in career_directions
                    if normalize_text(x)
                ],
            }
        )

    return cleaned


# ============================================================
# Domain retrieval helpers
# ============================================================

def path_filters(profile: Dict[str, Any]) -> Dict[str, str]:
    filters = {}

    city = normalize_text(profile.get("preferred_city"))
    if city:
        filters["city"] = city

    return filters


def find_colleges(
    path: str,
    profile: Dict[str, Any],
    indexes: Dict[str, Any],
    documents: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    query = (
        f"{path}. Student education stage: Matric. "
        f"Subjects: {profile.get('subjects', '')}. "
        f"Preferred city: {profile.get('preferred_city', '')}."
    )

    return retrieve_chunks(
        query,
        "colleges",
        indexes,
        documents,
        top_k=TOP_K,
        filters=path_filters(profile),
    )


def find_universities(
    degree: str,
    profile: Dict[str, Any],
    indexes: Dict[str, Any],
    documents: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    query = (
        f"Degree/program: {degree}. "
        f"Subjects: {profile.get('subjects', '')}. "
        f"Marks: {profile.get('marks', '')}. "
        f"Preferred city: {profile.get('preferred_city', '')}."
    )

    return retrieve_chunks(
        query,
        "universities",
        indexes,
        documents,
        top_k=TOP_K,
        filters=path_filters(profile),
    )


def find_scholarships(
    profile: Dict[str, Any],
    selected_institution: str,
    selected_program: str,
    indexes: Dict[str, Any],
    documents: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    query = (
        f"Education level: {profile.get('education_stage', '')}. "
        f"Institution: {selected_institution}. "
        f"Program: {selected_program}. "
        f"Marks/CGPA: {profile.get('marks', '')}. "
        f"Financial information/budget: {profile.get('budget', '')}."
    )

    return retrieve_chunks(
        query,
        "scholarships",
        indexes,
        documents,
        top_k=TOP_K,
    )


def find_careers(
    degree: str,
    indexes: Dict[str, Any],
    documents: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    query = f"Degree or study field: {degree}. Careers, jobs, skills, salaries, Pakistan, remote work, international opportunities."

    return retrieve_chunks(
        query,
        "careers",
        indexes,
        documents,
        top_k=TOP_K,
    )


# ============================================================
# Display helpers
# ============================================================

def display_retrieved_records(
    results: List[Dict[str, Any]],
    empty_message: str,
) -> None:
    if not results:
        st.info(empty_message)
        return

    for i, item in enumerate(results, start=1):
        title = item.get("institution") or item.get("program") or f"Result {i}"

        with st.expander(f"{i}. {title}", expanded=(i == 1)):
            st.write(item.get("text", ""))

            source = normalize_text(item.get("source"))
            if source:
                st.markdown(f"**Source:** {source}")


def clean_source_url(url: str) -> str:
    """Only return a URL if it looks like an HTTP(S) link."""
    url = normalize_text(url)
    if re.match(r"^https?://", url, flags=re.I):
        return url
    return ""


# ============================================================
# Expense module
# ============================================================

def parse_number(value: Any) -> Optional[float]:
    """Extract a simple numeric value from common fee strings."""
    text = normalize_text(value)
    if not text:
        return None

    # Avoid accidentally interpreting years as fees.
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)

    if not numbers:
        return None

    cleaned = numbers[0].replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def calculate_expenses(
    row_like_text: str,
) -> Dict[str, Any]:
    """
    Conservative expense calculator.
    It only calculates when clearly labelled numeric values can be found.
    It does not invent missing costs.
    """
    result: Dict[str, Any] = {
        "items": [],
        "total": None,
        "message": "",
    }

    labels = {
        "tuition": ["tuition fee", "tuition", "semester fee", "fee per semester"],
        "admission": ["admission fee", "admission"],
        "transport": ["transport", "transport fee"],
        "hostel": ["hostel", "hostel fee"],
        "other": ["other known costs", "other cost"],
    }

    found = []

    for label, aliases in labels.items():
        pattern = r"(?i)(?:"
        pattern += "|".join(re.escape(a) for a in aliases)
        pattern += r")\s*[:\-]?\s*(?:rs\.?|pkr)?\s*([\d,]+(?:\.\d+)?)"

        match = re.search(pattern, row_like_text)
        if match:
            try:
                amount = float(match.group(1).replace(",", ""))
                found.append((label, amount))
            except ValueError:
                pass

    if not found:
        result["message"] = (
            "Expense information is not available in a clearly calculable format "
            "in the current dataset."
        )
        return result

    total = sum(amount for _, amount in found)

    for label, amount in found:
        result["items"].append({"item": label.title(), "amount": amount})

    result["total"] = total
    return result


# ============================================================
# Roadmap generation
# ============================================================

def context_block(results: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    """Keep retrieved context bounded before sending it to Groq."""
    blocks = []
    current = 0

    for i, item in enumerate(results, start=1):
        text = normalize_text(item.get("text"))
        if not text:
            continue

        block = (
            f"[Retrieved record {i}]\n"
            f"{text}\n"
            f"Source: {normalize_text(item.get('source'))}\n"
        )

        if current + len(block) > max_chars:
            break

        blocks.append(block)
        current += len(block)

    return "\n".join(blocks)


def generate_roadmap(
    profile: Dict[str, Any],
    selected_path: str,
    institution_results: List[Dict[str, Any]],
    scholarship_results: List[Dict[str, Any]],
    career_results: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Generate a personalized explanation while keeping factual claims grounded
    in retrieved context.
    """
    system = GROUNDED_SYSTEM_PROMPT + """
Generate a practical student roadmap.

Use only the supplied retrieved records for factual claims about institutions,
programs, scholarships, fees, careers and salaries.

If a fact is not present, say it is not available in the current dataset.
Do not manufacture a complete list from general knowledge.

Use these headings:
1. Your Current Position
2. Recommended Academic Direction
3. Suggested Program / Path
4. Suitable Institutions
5. Estimated Expense
6. Scholarship Options
7. Skills to Build
8. Future Jobs
9. Long-Term Career Direction

Keep the roadmap concise and student-friendly.
"""

    user = f"""
Student profile:
{profile_to_text(profile)}

Selected academic path:
{selected_path}

INSTITUTION / PROGRAM CONTEXT:
{context_block(institution_results)}

SCHOLARSHIP CONTEXT:
{context_block(scholarship_results)}

CAREER CONTEXT:
{context_block(career_results)}

Create the personalized roadmap now.
"""

    return call_groq(system, user, temperature=0.25)


# ============================================================
# Main application
# ============================================================

def main() -> None:
    data = load_all_data()
    data_status(data)

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    st.info(
        "Career Compass Pakistan provides educational guidance based on the "
        "information you provide and the available dataset. Always verify "
        "admission, fee and scholarship details from official sources."
    )

    # Check whether core datasets are available.
    missing = [
        name for name, df in data.items()
        if df.empty
    ]

    if missing:
        st.warning(
            "Some datasets are missing or empty: "
            + ", ".join(missing)
            + ". Upload the corresponding CSV files into the data/ folder."
        )

    indexes, documents = get_indexes_and_documents(data)

    pages = [
        "🏠 Home",
        "👤 Student Assessment",
        "🎯 Recommendations",
        "🎓 Education Explorer",
        "💰 Scholarships & Expenses",
        "💼 Career Opportunities",
        "🗺 My Roadmap",
    ]

    page = st.sidebar.radio("Navigate", pages)

    # -------------------------
    # Home
    # -------------------------
    if page == "🏠 Home":
        st.header("Welcome to Career Compass Pakistan")
        st.write(
            "A simple AI + RAG platform that connects a student's profile "
            "to education options, scholarships, expenses and careers."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Data domains", 4)
            st.caption("Colleges, universities, scholarships and careers")

        with col2:
            st.metric("Retrieval", "FAISS")
            st.caption("Cosine-similarity semantic search")

        with col3:
            st.metric("AI layer", "Groq")
            st.caption("Profile reasoning and roadmap generation")

        st.markdown("### How it works")
        st.markdown(
            """
            **Student profile → AI-guided recommendations → RAG retrieval → "
            "education options → scholarships → expenses → careers → roadmap**
            """
        )

        st.markdown("### Accuracy principle")
        st.write(
            "The AI explains and personalizes the results, but the CSV datasets "
            "remain the source of truth for factual database information."
        )

        if get_groq_client() is None:
            st.warning(
                "AI features are currently disabled because GROQ_API_KEY is not "
                "available in Streamlit Secrets."
            )

    # -------------------------
    # Assessment
    # -------------------------
    elif page == "👤 Student Assessment":
        profile = create_student_profile()

        if profile:
            st.success("Student profile saved.")
            with st.expander("View saved profile"):
                st.json(profile)

    # -------------------------
    # Recommendations
    # -------------------------
    elif page == "🎯 Recommendations":
        profile = st.session_state.student_profile

        if not profile:
            st.info("Complete the Student Assessment first.")
            return

        st.header("🎯 AI-Guided Recommendations")

        if st.button(
            "Generate / refresh recommendations",
            type="primary",
        ):
            with st.spinner("Analyzing your profile..."):
                recommendations = get_ai_recommendations(profile)

            if recommendations:
                st.session_state.ai_recommendations = recommendations
            else:
                st.warning(
                    "No AI recommendations were generated. Check your Groq "
                    "configuration and try again."
                )

        recommendations = st.session_state.ai_recommendations

        if not recommendations:
            st.info("Click the button above to generate your top 3 recommendations.")
        else:
            for i, rec in enumerate(recommendations, start=1):
                with st.container(border=True):
                    st.subheader(f"{i}. {rec['name']}")
                    st.write(f"**Recommended fit:** {rec['fit']}")

                    if rec["future_fields"]:
                        st.write("**Possible future fields:**")
                        st.write(" • " + "\n • ".join(rec["future_fields"]))

                    if rec["career_directions"]:
                        st.write("**General career directions:**")
                        st.write(" • " + "\n • ".join(rec["career_directions"]))

                    if st.button(
                        f"Select {rec['name']}",
                        key=f"select_path_{i}",
                    ):
                        st.session_state.selected_path = rec["name"]
                        st.success(
                            f"Selected path: {st.session_state.selected_path}"
                        )

    # -------------------------
    # Education explorer
    # -------------------------
    elif page == "🎓 Education Explorer":
        profile = st.session_state.student_profile
        selected_path = st.session_state.selected_path

        if not profile:
            st.info("Complete the Student Assessment first.")
            return

        st.header("🎓 Education Explorer")

        if not selected_path:
            st.info("Select a recommendation first.")
            return

        st.success(f"Selected path: **{selected_path}**")

        if profile.get("education_stage") == "Matric":
            st.subheader("Suitable colleges / Intermediate options")

            if st.button("🔎 Find matching colleges", type="primary"):
                with st.spinner("Searching college data..."):
                    results = find_colleges(
                        selected_path,
                        profile,
                        indexes,
                        documents,
                    )

                st.session_state.college_results = results

            results = st.session_state.get("college_results", [])

            display_retrieved_records(
                results,
                "No matching colleges were found in the current dataset.",
            )

            if results:
                names = [
                    r.get("institution")
                    or r.get("program")
                    or f"College result {i+1}"
                    for i, r in enumerate(results)
                ]

                selected = st.selectbox(
                    "Select an institution/result",
                    names,
                )

                selected_item = next(
                    (
                        r for r in results
                        if (r.get("institution") or r.get("program")) == selected
                    ),
                    None,
                )

                if selected_item:
                    st.session_state.selected_institution = (
                        selected_item.get("institution") or selected
                    )
                    st.session_state.selected_program = (
                        selected_item.get("program") or selected_path
                    )

        else:
            st.subheader("Suitable Punjab universities / programs")

            if st.button("🔎 Find matching universities", type="primary"):
                with st.spinner("Searching university data..."):
                    results = find_universities(
                        selected_path,
                        profile,
                        indexes,
                        documents,
                    )

                st.session_state.university_results = results

            results = st.session_state.get("university_results", [])

            display_retrieved_records(
                results,
                "No matching universities were found in the current dataset.",
            )

            if results:
                options = [
                    r.get("institution")
                    or r.get("program")
                    or f"University result {i+1}"
                    for i, r in enumerate(results)
                ]

                selected = st.selectbox(
                    "Select an institution/result",
                    options,
                )

                selected_item = next(
                    (
                        r for r in results
                        if (r.get("institution") or r.get("program")) == selected
                    ),
                    None,
                )

                if selected_item:
                    st.session_state.selected_institution = (
                        selected_item.get("institution") or selected
                    )
                    st.session_state.selected_program = (
                        selected_item.get("program") or selected_path
                    )

        if st.session_state.selected_institution:
            st.success(
                "Selected institution: "
                + st.session_state.selected_institution
            )

    # -------------------------
    # Scholarships + expenses
    # -------------------------
    elif page == "💰 Scholarships & Expenses":
        profile = st.session_state.student_profile
        institution = st.session_state.selected_institution
        program = st.session_state.selected_program

        if not profile:
            st.info("Complete the Student Assessment first.")
            return

        if not institution or not program:
            st.info(
                "Select an institution/program in Education Explorer first."
            )
            return

        st.header("💰 Scholarships & Expenses")

        st.write(f"**Institution:** {institution}")
        st.write(f"**Program/path:** {program}")

        if st.button("🔎 Find scholarships", type="primary"):
            with st.spinner("Searching scholarship data..."):
                scholarships = find_scholarships(
                    profile,
                    institution,
                    program,
                    indexes,
                    documents,
                )

            st.session_state.retrieved_scholarships = scholarships

        scholarships = st.session_state.retrieved_scholarships

        display_retrieved_records(
            scholarships,
            "No relevant scholarships were found in the current dataset.",
        )

        if scholarships:
            st.caption(
                "You may be eligible based on the available criteria. "
                "Please verify the latest requirements from the official source."
            )

            for item in scholarships:
                source = clean_source_url(item.get("source", ""))
                if source:
                    st.markdown(f"[Official/source link]({source})")

        st.divider()
        st.subheader("Estimated expenses")

        # Use retrieved education records rather than inventing values.
        education_results = (
            st.session_state.get("university_results", [])
            + st.session_state.get("college_results", [])
        )

        selected_record = None
        for item in education_results:
            if (
                normalize_text(item.get("institution")) == institution
                or normalize_text(item.get("program")) == program
            ):
                selected_record = item
                break

        if selected_record:
            expense_text = selected_record.get("text", "")
            expense = calculate_expenses(expense_text)

            if expense["items"]:
                for item in expense["items"]:
                    st.write(
                        f"**{item['item']}:** "
                        f"Rs {item['amount']:,.0f} (Estimated)"
                    )

                st.metric(
                    "Estimated total of clearly identified costs",
                    f"Rs {expense['total']:,.0f}",
                )
            else:
                st.info(expense["message"])
        else:
            st.info(
                "Expense information is not available in the current dataset."
            )

    # -------------------------
    # Careers
    # -------------------------
    elif page == "💼 Career Opportunities":
        profile = st.session_state.student_profile
        selected_program = st.session_state.selected_program
        selected_path = st.session_state.selected_path

        if not profile:
            st.info("Complete the Student Assessment first.")
            return

        degree = selected_program or selected_path

        if not degree:
            st.info("Select a study path or program first.")
            return

        st.header("💼 Career Opportunities")
        st.write(f"Searching career data for: **{degree}**")

        if st.button("🔎 Find careers", type="primary"):
            with st.spinner("Searching career data..."):
                careers = find_careers(
                    degree,
                    indexes,
                    documents,
                )

            st.session_state.retrieved_careers = careers

        careers = st.session_state.retrieved_careers

        display_retrieved_records(
            careers,
            "No relevant careers were found in the current dataset.",
        )

        if careers:
            st.caption(
                "Salary estimates vary by experience, employer, city, economic "
                "conditions, and market demand."
            )

    # -------------------------
    # Roadmap
    # -------------------------
    elif page == "🗺 My Roadmap":
        profile = st.session_state.student_profile
        selected_path = st.session_state.selected_path

        if not profile:
            st.info("Complete the Student Assessment first.")
            return

        if not selected_path:
            st.info("Select an academic path first.")
            return

        st.header("🗺 My Roadmap")

        institution_results = (
            st.session_state.get("university_results", [])
            + st.session_state.get("college_results", [])
        )
        scholarship_results = st.session_state.retrieved_scholarships
        career_results = st.session_state.retrieved_careers

        if st.button(
            "✨ Generate personalized roadmap",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Creating your grounded roadmap..."):
                roadmap = generate_roadmap(
                    profile,
                    selected_path,
                    institution_results,
                    scholarship_results,
                    career_results,
                )

            if roadmap:
                st.session_state.roadmap = roadmap

        if st.session_state.roadmap:
            st.markdown(st.session_state.roadmap)
        else:
            st.info(
                "For the best roadmap, first select a path, explore institutions, "
                "retrieve scholarships and retrieve careers."
            )

        st.divider()
        st.caption(
            "Reminder: this is educational guidance based on your profile and "
            "the available dataset. Verify current admission, fee and scholarship "
            "information from official sources."
        )


if __name__ == "__main__":
    main()
