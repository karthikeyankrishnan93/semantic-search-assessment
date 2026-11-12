import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

sentences = [
    "Banks must verify customer identity before opening accounts.",
    "Before opening an account, banks must confirm each customer's identity.",
    "Banks should monitor suspicious financial transactions.",
    "Financial institutions must maintain accurate transaction records.",
    "Employees must use strong passwords to protect company systems.",
    "Multi-factor authentication improves information security.",
    "Sensitive customer data must not be shared with outsiders.",
    "Employees must submit leave requests through the HR system.",
    "New employees must complete the company onboarding process.",
    "The HR department organizes employee training programs."
]

categories = [
    "Banking", "Banking", "Banking", "Banking",
    "Security", "Security", "Security",
    "HR", "HR", "HR"
]

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)

print("\n========== COSINE FUNCTION TESTS ==========")

test_cases = [
    ([1, 2, 3], [2, 4, 6], 1.0),
    ([1, 0], [0, 1], 0.0),
    ([1, 0], [-1, 0], -1.0),
]

for a, b, expected in test_cases:
    actual = cosine_similarity(a, b)
    print(f"{a} vs {b} -> {actual:.1f}")
    assert abs(actual - expected) < 1e-6

print("\n========== LOCAL EMBEDDINGS ==========")

local_model = SentenceTransformer("all-MiniLM-L6-v2")

start_time = time.perf_counter()

local_embeddings = local_model.encode(
    sentences,
    convert_to_numpy=True
)

local_time = time.perf_counter() - start_time

print("Number of sentences:", len(sentences))
print("Local embedding shape:", local_embeddings.shape)
print(f"Local embedding time: {local_time:.4f} seconds")

print("\n========== GEMINI API EMBEDDINGS ==========")

api_key = os.getenv("GEMINI_API_KEY")

gemini_embeddings = None
gemini_time = None

if api_key and api_key != "your_gemini_api_key_here":

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        start_time = time.perf_counter()

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=sentences,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
        )

        gemini_time = time.perf_counter() - start_time

        gemini_embeddings = np.array(
            [item.values for item in response.embeddings]
        )

        print("Gemini embedding shape:", gemini_embeddings.shape)
        print(f"Gemini API time: {gemini_time:.4f} seconds")

    except Exception as error:
        print("Gemini API failed:", error)

else:
    print("Gemini API skipped: API key not configured.")

print("\n========== EXECUTION TIME COMPARISON ==========")

print(f"Local Model: {local_time:.4f} seconds")

if gemini_time is not None:
    print(f"Gemini API: {gemini_time:.4f} seconds")
else:
    print("Gemini API: Not measured (API key unavailable)")

print("\n========== SIMILARITY MATRIX ==========")

similarity_matrix = np.zeros(
    (len(sentences), len(sentences))
)

for i in range(len(sentences)):
    for j in range(len(sentences)):
        similarity_matrix[i][j] = cosine_similarity(
            local_embeddings[i],
            local_embeddings[j]
        )

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"S{i+1}" for i in range(len(sentences))],
    columns=[f"S{i+1}" for i in range(len(sentences))]
)

print(similarity_df.round(3))

similarity_df.to_csv(
    OUTPUT_DIR / "similarity_matrix.csv"
)

print("\n========== SCIKIT-LEARN COMPARISON ==========")

sklearn_matrix = sklearn_cosine(local_embeddings)

difference = np.max(
    np.abs(similarity_matrix - sklearn_matrix)
)

print("Maximum difference:", difference)

assert np.allclose(
    similarity_matrix,
    sklearn_matrix,
    atol=1e-6
)

print("Our cosine function matches scikit-learn.")

print("\n========== TOP THREE SIMILAR PAIRS ==========")

pairs = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append(
            (similarity_matrix[i][j], i, j)
        )

pairs.sort(reverse=True)

for score, i, j in pairs[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print("Sentence 1:", sentences[i])
    print("Sentence 2:", sentences[j])

print("\n========== NO COMMON WORDS EXAMPLE ==========")

sentence_a = "Physician treated patient."
sentence_b = "Doctor cared for sick person."

embedding_a = local_model.encode(sentence_a)
embedding_b = local_model.encode(sentence_b)

common_words = (
    set(sentence_a.lower().replace(".", "").split())
    &
    set(sentence_b.lower().replace(".", "").split())
)

score = cosine_similarity(embedding_a, embedding_b)

print("Sentence 1:", sentence_a)
print("Sentence 2:", sentence_b)
print("Common words:", common_words)
print(f"Semantic similarity: {score:.4f}")

print("\n========== SEMANTIC SEARCH ==========")

queries = [
    "How can a bank identify its customers?",
    "How can company systems be protected?",
    "How do employees apply for leave?"
]

for query in queries:
    query_embedding = local_model.encode(query)

    scores = [
        cosine_similarity(query_embedding, embedding)
        for embedding in local_embeddings
    ]

    best_index = int(np.argmax(scores))

    print("\nQuery:", query)
    print("Best match:", sentences[best_index])
    print(f"Similarity: {scores[best_index]:.4f}")

print("\n========== DUPLICATE DETECTION ==========")

duplicate_threshold = 0.75
duplicates_found = False

for score, i, j in pairs:
    if score >= duplicate_threshold:
        duplicates_found = True

        print(f"\nPossible duplicate: {score:.4f}")
        print("Sentence 1:", sentences[i])
        print("Sentence 2:", sentences[j])

if not duplicates_found:
    print("No duplicate pairs found above threshold.")

print("Duplicate threshold:", duplicate_threshold)

print("\n========== K-MEANS CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(local_embeddings)

cluster_names = {}

for cluster_id in range(3):
    members = [
        categories[i]
        for i in range(len(categories))
        if cluster_labels[i] == cluster_id
    ]

    cluster_names[cluster_id] = max(
        set(members),
        key=members.count
    )

for cluster_id in range(3):
    print(f"\nCluster {cluster_id + 1}: {cluster_names[cluster_id]}")

    for index, label in enumerate(cluster_labels):
        if label == cluster_id:
            print("-", sentences[index])

print("\n========== MODEL SELECTION ==========")

print("""
Local Model:
1. No API cost for local inference.
2. Works without internet after model download.
3. Suitable for development and offline applications.
4. Requires local memory and processing power.

API Model:
1. Easy to access through an API.
2. No need to manage the embedding model locally.
3. Requires internet and API access.
4. Usage limits and pricing must be considered.

Decision:
Choose a local model for offline use and predictable
local processing. Choose an API model when managed
infrastructure and API-based integration are preferred.
""")
