import csv
import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from chunking_utils import (
    extract_all_documents,
    fixed_token_chunking,
    sentence_chunking,
    recursive_chunking,
    build_chunks,
    describe_chunk
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

docs = extract_all_documents(DATA_DIR)

if not docs:
    raise ValueError("No documents found in data folder.")

print("\n1. DOCUMENT EXTRACTION")
print("Documents:", len(set(d["source"] for d in docs)))
print("Pages/text units:", len(docs))

full_text = "\n\n".join(d["text"] for d in docs)

fixed = fixed_token_chunking(full_text, 512, 50)
sentence = sentence_chunking(full_text, 3)
recursive = recursive_chunking(full_text, 512, 50)

print("\n2. CHUNKING COMPARISON")
print("Fixed token:", len(fixed))
print("Sentence based:", len(sentence))
print("Recursive:", len(recursive))

small_chunks = fixed_token_chunking(docs[0]["text"], 22, 3)

print("\nSENTENCE CUT DEMONSTRATION")
for i, chunk in enumerate(small_chunks[:2], 1):
    print(f"Chunk {i}:", repr(chunk[-100:]))

questions = [
    ("Who communicates policy changes?", "policy.pdf", 1),
    ("How are leave requests submitted?", "policy.pdf", 2),
    ("Where are conflicts of interest reported?", "policy.pdf", 3),
    ("Who receives access to company systems?", "security_manual.pdf", 1),
    ("Where is password compromise reported?", "security_manual.pdf", 2),
    ("How is customer information protected?", "security_manual.pdf", 3),
    ("What does employee onboarding include?", "employee_manual.docx", 1),
    ("How is employee attendance recorded?", "employee_manual.docx", 1),
    ("When is customer identity verified?", "compliance_manual.docx", 1),
    ("Where are suspicious transactions escalated?", "banking_policy.txt", 1)
]

print("\n3. CHUNK SIZE EXPERIMENT")

model = SentenceTransformer("all-MiniLM-L6-v2")
question_vectors = model.encode(
    [q[0] for q in questions],
    normalize_embeddings=True
)

experiment_results = []

for size in [128, 256, 512, 1024]:
    chunks = []

    for doc in docs:
        for index, content in enumerate(
            fixed_token_chunking(doc["text"], size, max(1, size // 10))
        ):
            chunks.append({
                "text": content,
                "source": doc["source"],
                "page": doc["page"],
                "index": index
            })

    if not chunks:
        continue

    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True
    )

    scores = question_vectors @ vectors.T
    hit1 = hit3 = 0

    for i, question in enumerate(questions):
        top = np.argsort(scores[i])[::-1][:3]

        matches = [
            chunks[j]["source"] == question[1]
            and chunks[j]["page"] == question[2]
            for j in top
        ]

        hit1 += int(matches[0])
        hit3 += int(any(matches))

    result = {
        "chunk_size": size,
        "total_chunks": len(chunks),
        "hit_at_1": round(hit1 / len(questions), 4),
        "hit_at_3": round(hit3 / len(questions), 4)
    }

    experiment_results.append(result)
    print(result)

with open(OUTPUT_DIR / "chunk_size_experiment.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=experiment_results[0].keys())
    writer.writeheader()
    writer.writerows(experiment_results)

print("\n4. METADATA AND CITATIONS")

metadata_chunks = build_chunks(docs, 512, 50)

with open(OUTPUT_DIR / "metadata_chunks.jsonl", "w", encoding="utf-8") as file:
    for chunk in metadata_chunks:
        file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

for chunk in metadata_chunks[:3]:
    print("ID:", chunk["id"])
    print("Citation:", describe_chunk(chunk))
    print("Metadata:", chunk["metadata"])

print("\n5. STABLE ID AND UPSERT TEST")

vector_store = {}

def upsert(chunks):
    for chunk in chunks:
        vector_store[chunk["id"]] = chunk

upsert(metadata_chunks)
first_count = len(vector_store)

upsert(metadata_chunks)
second_count = len(vector_store)

print("First indexing:", first_count)
print("Second indexing:", second_count)
print("No duplicate growth:", first_count == second_count)

print("\n6. INCREMENTAL UPDATE")

edited_docs = [dict(doc) for doc in docs]

target = next(
    d for d in edited_docs
    if d["source"] == "policy.pdf" and d["page"] == 1
)

lines = target["text"].splitlines()

for i, line in enumerate(lines):
    if len(line.strip()) >= 40:
        lines[i] = (
            line
            + " Updated paragraph for incremental indexing verification."
        )
        break
else:
    raise ValueError("No suitable paragraph found.")

target["text"] = "\n".join(lines)

updated_chunks = build_chunks(edited_docs, 512, 50)

old_page = {
    c["id"] for c in metadata_chunks
    if c["metadata"]["source"] == "policy.pdf"
    and c["metadata"]["page"] == 1
}

new_page = {
    c["id"] for c in updated_chunks
    if c["metadata"]["source"] == "policy.pdf"
    and c["metadata"]["page"] == 1
}

removed_ids = old_page - new_page
added_ids = new_page - old_page

print("Edited document: policy.pdf")
print("Edited page: 1")
print("Removed old chunk IDs:", len(removed_ids))
print("Added new chunk IDs:", len(added_ids))

def other_page_ids(chunks):
    return {
        c["id"]
        for c in chunks
        if not (
            c["metadata"]["source"] == "policy.pdf"
            and c["metadata"]["page"] == 1
        )
    }

unchanged = (
    other_page_ids(metadata_chunks)
    == other_page_ids(updated_chunks)
)

for chunk_id in list(vector_store):
    chunk = vector_store[chunk_id]
    if (
        chunk["metadata"]["source"] == "policy.pdf"
        and chunk["metadata"]["page"] == 1
    ):
        del vector_store[chunk_id]

updated_page_chunks = [
    c for c in updated_chunks
    if c["metadata"]["source"] == "policy.pdf"
    and c["metadata"]["page"] == 1
]

upsert(updated_page_chunks)

print("Other pages unchanged:", unchanged)
print("Updated page chunks:", len(updated_page_chunks))
print("Final index count:", len(vector_store))

print("\nAll experiments completed. Check outputs folder.")