# Semantic Document Search

## 1. Project Overview

This project implements a semantic document search system using embeddings, vector databases, and FastAPI.

The system extracts text from PDF, DOCX, and TXT documents, splits the content into chunks, generates embeddings, and stores them in Qdrant. Users can search documents using natural language queries and retrieve relevant content with source and page information.

## 2. Technologies Used

- Python 3.12
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Google Gemini Embedding API
- NumPy and Scikit-learn
- FAISS
- Qdrant
- FastAPI
- PyMuPDF
- python-docx
- LangChain Text Splitters
- Pytest

## 3. Project Structure

semantic_search/
│
├── data/
│   ├── evaluation/
│   │   └── evaluation_queries.json
│   ├── uploaded_documents/
│   ├── policy.pdf
│   ├── security_manual.pdf
│   ├── employee_manual.docx
│   ├── compliance_manual.docx
│   └── banking_policy.txt
│
├── outputs/
│   ├── chunk_size_experiment.csv
│   ├── metadata_chunks.jsonl
│   ├── similarity_matrix.csv
│   └── retrieval_evaluation_report.txt
│
├── scripts/
│   └── create_sample_docs.py
│
├── src/
│   ├── task1_embeddings.py
│   ├── chunking_utils.py
│   ├── task2_chunking.py
│   ├── task3_vectordb.py
│   ├── api.py
│   └── evaluate_search.py
│
├── tests/
│   └── test_api.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md


## 4. Assessment Tasks and Results

### Task 1 – Embeddings and Cosine Similarity

Implemented:
- Generated embeddings using local Sentence Transformers and Google Gemini.
- Compared embedding dimensions and execution time.
- Implemented cosine similarity from scratch.
- Verified cosine similarity with Scikit-learn.
- Generated a similarity matrix.
- Identified similar sentence pairs.
- Demonstrated semantic search, duplicate detection, and K-Means clustering.

**Actual results:**
- Local embedding model: `all-MiniLM-L6-v2`
- Local embedding dimension: 384
- Gemini embedding dimension: 768
- Number of sentences tested: 10
- Cosine similarity test cases: 3 passed.

### Task 2 – Chunking, Metadata and Stable IDs

Implemented:
- Extracted content from PDF, DOCX, and TXT documents.
- Implemented fixed-token, sentence-based, and recursive chunking.
- Compared different chunk sizes.
- Added metadata to every chunk.
- Generated stable UUID-based chunk IDs.
- Implemented incremental updates and duplicate prevention.
- Generated source, page, and section citations.

**Actual results:**

| Chunking Method | Number of Chunks |
|---|---:|
| Fixed-token | 2 |
| Sentence-based | 17 |
| Recursive | 12 |

**Chunk-size experiment:**

| Chunk Size | Total Chunks | Hit@1 | Hit@3 |
|---|---:|---:|---:|
| 128 | 10 | 100% | 100% |
| 256 | 9 | 100% | 100% |
| 512 | 9 | 100% | 100% |
| 1024 | 9 | 100% | 100% |

Stable ID and upsert verification:
- First indexing: 12 chunks.
- Second indexing: 12 chunks.
- Duplicate growth: None.
- Incremental update: Only the changed page's chunk was replaced.

### Task 3 – Vector Storage and Approximate Nearest Neighbour Search

Implemented:
- Compared exact search and FAISS HNSW search.
- Explained Approximate Nearest Neighbour (ANN) and HNSW.
- Created a Qdrant collection using cosine similarity.
- Stored document embeddings and metadata.
- Configured HNSW indexing.
- Implemented metadata-filtered search.
- Verified repeated upserts and document deletion.

**FAISS experiment – Latest run:**

| Metric | Result |
|---|---:|
| Dataset | 50,000 vectors |
| Dimensions | 128 |
| Exact search time | 55.74 ms |
| HNSW search time | 16.47 ms |
| Recall@10 | 0.6120 |

**Qdrant results:**
- Collection: `semantic_document_vectors`
- Initial points: 12
- Points after repeated upsert: 12
- Points after deleting one source: 10
- Distance metric: Cosine
- HNSW configuration: m=16, ef_construct=100
- Final collection status: Green
- Index verification: Qdrant collection status was Green, optimizer status was OK, and indexed vector status was checked after indexing.


### Task 4 – Search API and Retrieval Evaluation

Implemented a FastAPI-based semantic document search application.

**API endpoints:**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/search` | Search documents |
| GET | `/documents` | List indexed documents |
| POST | `/documents` | Upload and index documents |

Additional features:
- Loads the embedding model once during application startup.
- API key authentication using `x-api-key`.
- Validates `top_k` between 1 and 20.
- Supports document-type and minimum-score filters.
- Returns ranked results with source, page, section, score, and text.
- Supports PDF, DOCX, and TXT uploads.

**Retrieval evaluation:**

Tested 15 evaluation queries against expected source documents and pages.

| Metric | Actual Result |
|---|---:|
| Total queries | 15 |
| Hit@1 | 86.67% |
| Hit@3 | 100% |
| Hit@5 | 100% |
| MRR | 0.9222 |

**Configuration comparison:**

Compared chunk sizes 128 and 512.

Both achieved 100% Hit@1 and Hit@3 in the chunk-size experiment. Chunk size 512 generated fewer chunks (9 compared with 10), so it was preferred for this small dataset.

**Automated testing:**

- Total pytest tests: 6
- Passed: 6
- Failed: 0

## 5. Setup Instructions

### Step 1 – Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2 – Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3 – Configure environment variables

Create `.env` using `.env.example` as a reference.

Configure the required API keys and application settings. Do not upload `.env` or expose API keys.

### Step 4 – Start Qdrant

Start the configured Qdrant Docker container:

```powershell
docker start semantic-qdrant
```

### Step 5 – Run the assessment scripts

```powershell
python src/task1_embeddings.py
python src/task2_chunking.py
python src/task3_vectordb.py
```

### Step 6 – Start the Search API

```powershell
uvicorn src.api:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### Step 7 – Run retrieval evaluation

Keep the API running and use another terminal:

```powershell
python src/evaluate_search.py
```

### Step 8 – Run automated tests

```powershell
python -m pytest -v
```

## 6. Important Output Files

| File | Description |
|---|---|
| `outputs/similarity_matrix.csv` | Sentence similarity results |
| `outputs/chunk_size_experiment.csv` | Chunk-size comparison |
| `outputs/metadata_chunks.jsonl` | Chunk content, metadata, and stable IDs |
| `outputs/retrieval_evaluation_report.txt` | Search evaluation metrics and configuration comparison |

## 7. Vector Database Comparison

| Feature | pgvector | Qdrant |
|---|---|---|
| Setup | PostgreSQL extension | Dedicated vector database |
| Filtering | SQL queries | Payload filters |
| Indexing | HNSW and IVFFlat | HNSW |
| Integration | PostgreSQL applications | Vector search applications |
| Suitable for | Applications already using PostgreSQL | Dedicated semantic search systems |

**Choice:** Qdrant was used for this project because it provides dedicated vector search, metadata filtering, and convenient integration with the FastAPI application.

## 8. Key Learnings

- Generating and comparing text embeddings.
- Understanding cosine similarity and semantic similarity.
- Extracting documents and implementing chunking strategies.
- Managing metadata, stable IDs, and incremental upserts.
- Understanding FAISS, ANN, and HNSW.
- Storing and retrieving vectors using Qdrant.
- Building authenticated search APIs with FastAPI.
- Evaluating retrieval quality using Hit@K and MRR.
- Writing automated tests using Pytest.

## 9. Final Project Status

All four assessment tasks have been implemented and tested. The reported metrics are based on the actual experiment outputs from this project.

**Project:** Semantic Document Search  
**Status:** Implementation and testing completed.