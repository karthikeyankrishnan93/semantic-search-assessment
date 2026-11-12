import json
import os
import secrets
import uuid
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Depends, UploadFile
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from src.chunking_utils import extract_document, build_chunks

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "metadata_chunks.jsonl"
UPLOAD_DIR = DATA_DIR / "uploaded_documents"

load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
APP_API_KEY = os.getenv("APP_API_KEY", "")

COLLECTION = "semantic_document_search"
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(
    title="Semantic Document Search API",
    description="Search banking and policy documents using semantic similarity.",
    version="1.0.0"
)

model = None
client = None

def verify_api_key(
    x_api_key: str = Header(default=None, alias="x-api-key")
):
    if not APP_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured."
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, APP_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )

    return True

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    min_score: float | None = Field(default=None, ge=-1, le=1)

def generate_embeddings(texts):
    return model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

def store_chunks(chunks):
    if not chunks:
        return 0

    vectors = generate_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True
    )

    return len(points)

@app.on_event("startup")
def startup_event():
    global model, client

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {DATA_FILE}"
        )

    with open(DATA_FILE, encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    stored = store_chunks(chunks)

    print("Model loaded successfully.")
    print("API collection:", COLLECTION)
    print("Chunks indexed:", stored)

@app.get("/health")
def health_check(authenticated: bool = Depends(verify_api_key)):
    if client is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Application is not ready."
        )

    info = client.get_collection(COLLECTION)

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "collection": COLLECTION,
        "points_count": info.points_count
    }

@app.post("/search")
def search_documents(
    request: SearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    if not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty."
        )

    query_vector = generate_embeddings(
        [request.query]
    )[0]

    search_filter = None

    if request.doc_type:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(
                        value=request.doc_type.upper()
                    )
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=request.top_k,
        score_threshold=request.min_score,
        with_payload=True
    ).points

    return {
        "query": request.query,
        "total_results": len(results),
        "results": [
            {
                "source": result.payload["source"],
                "page": result.payload["page"],
                "section": result.payload["section_heading"],
                "score": round(result.score, 4),
                "text": result.payload["text"]
            }
            for result in results
        ]
    }

@app.get("/documents")
def list_documents(
    authenticated: bool = Depends(verify_api_key)
):
    documents = {}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:
            source = point.payload["source"]

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "doc_type": point.payload["doc_type"],
                    "chunk_count": 0
                }

            documents[source]["chunk_count"] += 1

        if next_offset is None:
            break

        offset = next_offset

    return {
        "document_count": len(documents),
        "documents": list(documents.values())
    }

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are supported."
        )

    upload_folder = UPLOAD_DIR / str(uuid.uuid4())
    upload_folder.mkdir(parents=True, exist_ok=True)

    file_path = upload_folder / filename

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    file_path.write_bytes(content)

    try:
        documents = extract_document(file_path)
        chunks = build_chunks(documents)

        if not chunks:
            raise ValueError("No text could be extracted.")

        stored = store_chunks(chunks)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing failed: {error}"
        )

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": filename,
        "chunks_indexed": stored
    }