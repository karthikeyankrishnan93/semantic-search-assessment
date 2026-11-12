import json
import time
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "outputs" / "metadata_chunks.jsonl"

COLLECTION = "semantic_document_vectors"
MODEL_NAME = "all-MiniLM-L6-v2"


print("\n1. FAISS EXACT SEARCH VS HNSW")

np.random.seed(42)
dimension = 128
data = np.random.random((50000, dimension)).astype("float32")
queries = np.random.random((100, dimension)).astype("float32")

exact_index = faiss.IndexFlatL2(dimension)
exact_index.add(data)

hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
hnsw_index.hnsw.efConstruction = 100
hnsw_index.hnsw.efSearch = 64
hnsw_index.add(data)

start = time.perf_counter()
exact_distances, exact_ids = exact_index.search(queries, 10)
exact_time = (time.perf_counter() - start) * 1000

start = time.perf_counter()
hnsw_distances, hnsw_ids = hnsw_index.search(queries, 10)
hnsw_time = (time.perf_counter() - start) * 1000

recall = np.mean([
    len(set(exact_ids[i]) & set(hnsw_ids[i])) / 10
    for i in range(len(queries))
])

print(f"Vectors: {len(data)} | Dimension: {dimension}")
print(f"Exact search: {exact_time:.2f} ms")
print(f"HNSW search: {hnsw_time:.2f} ms")
print(f"Recall@10: {recall:.4f}")

print("\n2. LOAD CHUNKS AND GENERATE EMBEDDINGS")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing output: {DATA_FILE}")

with open(DATA_FILE, encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

if not chunks:
    raise ValueError("No metadata chunks found.")

model = SentenceTransformer(MODEL_NAME)
vectors = model.encode(
    [chunk["text"] for chunk in chunks],
    normalize_embeddings=True
)

print("Chunks:", len(chunks))
print("Embedding shape:", vectors.shape)

print("\n3. QDRANT STORAGE AND HNSW INDEX")

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=vectors.shape[1],
        distance=models.Distance.COSINE
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=1
    )
)

client.create_payload_index(
    collection_name=COLLECTION,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True
)

client.create_payload_index(
    collection_name=COLLECTION,
    field_name="page",
    field_schema=models.PayloadSchemaType.INTEGER,
    wait=True
)

client.create_payload_index(
    collection_name=COLLECTION,
    field_name="source",
    field_schema=models.PayloadSchemaType.KEYWORD,
    wait=True
)

points = [
    models.PointStruct(
        id=chunk["id"],
        vector=vectors[i].tolist(),
        payload={
            "text": chunk["text"],
            **chunk["metadata"]
        }
    )
    for i, chunk in enumerate(chunks)
]

client.upsert(
    collection_name=COLLECTION,
    points=points,
    wait=True
)

first_count = client.count(
    collection_name=COLLECTION,
    exact=True
).count

info = client.get_collection(COLLECTION)

print("Collection:", COLLECTION)
print("Stored points:", first_count)
print("Collection status:", info.status)
print("Indexed vectors:", info.indexed_vectors_count)
print("HNSW configuration:", info.config.hnsw_config)

print("\n4. FILTERED SEARCH")

query_vector = model.encode(
    "How are employee and customer policies protected?",
    normalize_embeddings=True
).tolist()

search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="PDF")
        ),
        models.FieldCondition(
            key="page",
            range=models.Range(gte=3)
        )
    ]
)

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    query_filter=search_filter,
    limit=5,
    with_payload=True
).points

print("Filter: doc_type=PDF, page >= 3")
print("Results:", len(results))

for result in results:
    print(
        result.payload["source"],
        "| Page:", result.payload["page"],
        "| Score:", round(result.score, 4)
    )

print("\n5. RE-UPSERT VERIFICATION")

client.upsert(
    collection_name=COLLECTION,
    points=points,
    wait=True
)

second_count = client.count(
    collection_name=COLLECTION,
    exact=True
).count

print("Before re-upsert:", first_count)
print("After re-upsert:", second_count)
print("No duplicate growth:", first_count == second_count)

print("\n6. DELETE BY SOURCE")

delete_source = "banking_policy.txt"

client.delete(
    collection_name=COLLECTION,
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=delete_source)
                )
            ]
        )
    ),
    wait=True
)

final_count = client.count(
    collection_name=COLLECTION,
    exact=True
).count

print("Deleted source:", delete_source)
print("Count before deletion:", second_count)
print("Count after deletion:", final_count)
