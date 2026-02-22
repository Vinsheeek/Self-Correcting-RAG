import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ✅ Load Embedding Model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = []

DATA_PATH = "data/"

# ✅ Load PDF & TXT Files
for file in os.listdir(DATA_PATH):

    path = os.path.join(DATA_PATH, file)

    if file.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        documents.append(text)

    elif file.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())

print("Documents Loaded:", len(documents))

# ✅ Generate Embeddings
embeddings = embedder.encode(documents)

# ✅ Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS Index Ready")

# ✅ Retrieval Function
def retrieve(query, k=2):

    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = [documents[i] for i in indices[0]]

    return results

# ✅ Simulated Generator
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

# ✅ Validator + Confidence Engine
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

# ✅ Quality Evaluator
def evaluate_answer_quality(answer):

    length_score = min(len(answer) / 300, 1.0)
    structure_score = 1.0 if "Based on retrieved context" in answer else 0.5

    quality_score = (length_score + structure_score) / 2

    return quality_score

# ✅ Self-Correction Engine
def self_correct(query):

    retrieved_docs = retrieve(query)
    answer = generate_answer(query, retrieved_docs)

    is_valid, reason, confidence = validate_response(query, retrieved_docs, answer)

    if not is_valid:

        print("\n⚠ Validation Failed:", reason)
        print("🔁 Retrying with expanded retrieval...\n")

        retrieved_docs = retrieve(query, k=4)
        answer = generate_answer(query, retrieved_docs)

        is_valid, reason, confidence = validate_response(query, retrieved_docs, answer)

    return retrieved_docs, answer, confidence

# ✅ Full Pipeline
def rag_pipeline(query):

    retrieved_docs, answer, confidence = self_correct(query)

    quality = evaluate_answer_quality(answer)

    return retrieved_docs, answer, confidence, quality

# ✅ Terminal Testing
if __name__ == "__main__":

    query = input("Enter your question: ")

    docs, answer, confidence, quality = rag_pipeline(query)

    print("\n============================")
    print("📄 Retrieved Context:\n")

    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---\n")
        print(doc[:500])

    print("\n============================")
    print("Generated Answer:\n")
    print(answer)

    print("\nConfidence Score:", round(confidence * 100, 2), "%")
    print("Answer Quality Score:", round(quality * 100, 2), "%")