import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Self-Correcting RAG", layout="wide")

st.title("🧠 Self-Correcting RAG Pipeline")
st.write("Upload documents and ask questions. System validates responses & computes confidence.")

# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_model()

# -----------------------------
# Validator
# -----------------------------
def validate_response(query, retrieved_docs, answer):

    if len(retrieved_docs) == 0:
        return False, "No documents retrieved", 0

    context = retrieved_docs[0].lower()
    query_terms = query.lower().split()

    matches = sum(term in context for term in query_terms)

    confidence = min(matches / len(query_terms), 1.0)

    if matches < 2:
        return False, "Weak retrieval context", confidence

    return True, "Response validated", confidence


# -----------------------------
# Quality Score
# -----------------------------
def evaluate_answer_quality(answer):

    length_score = min(len(answer) / 300, 1.0)
    structure_score = 1.0 if "Based on retrieved context" in answer else 0.5

    quality_score = (length_score + structure_score) / 2

    return quality_score


# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs or TXT Files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

documents = []

if uploaded_files:

    for file in uploaded_files:

        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()

            documents.append(text)

        elif file.name.endswith(".txt"):
            documents.append(file.read().decode("utf-8"))

    st.success(f"✅ Loaded {len(documents)} documents")

# -----------------------------
# Build Vector Index
# -----------------------------
if documents:

    with st.spinner("🔎 Creating embeddings & vector index..."):

        embeddings = embedder.encode(documents)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    st.success("✅ FAISS Index Ready")

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve(query, k=2):

        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), k)

        results = [documents[i] for i in indices[0]]

        return results

    # -----------------------------
    # Generator
    # -----------------------------
    def generate_answer(query, retrieved_docs):

        context = retrieved_docs[0][:200]

        answer = f"""
Simulated Response:

Based on retrieved context:
{context}

Answer to query:
{query}
"""

        return answer

    # -----------------------------
    # Self Correction
    # -----------------------------
    def self_correct(query):

        retrieved_docs = retrieve(query)
        answer = generate_answer(query, retrieved_docs)

        is_valid, reason, confidence = validate_response(query, retrieved_docs, answer)

        if not is_valid:

            retrieved_docs = retrieve(query, k=4)
            answer = generate_answer(query, retrieved_docs)

            is_valid, reason, confidence = validate_response(query, retrieved_docs, answer)

        return retrieved_docs, answer, confidence, reason, is_valid

    # -----------------------------
    # Query Input
    # -----------------------------
    query = st.text_input("❓ Ask your question")

    if query:

        retrieved_docs, answer, confidence, reason, is_valid = self_correct(query)
        quality = evaluate_answer_quality(answer)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Retrieved Context")

            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Document {i+1} Preview:**")
                st.write(doc[:300])
                st.write("---")

        with col2:
            st.subheader("💡 Generated Answer")
            st.text(answer)

            if is_valid:
                st.success("✅ Response Validated")
            else:
                st.warning(f"⚠ {reason}")

            st.metric("Confidence Score", f"{round(confidence * 100, 2)} %")
            st.metric("Answer Quality Score", f"{round(quality * 100, 2)} %")

else:
    st.info("👆 Upload documents to begin")