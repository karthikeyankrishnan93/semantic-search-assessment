import re
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pymupdf
from docx import Document
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

NAMESPACE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")

def get_file_created_date(file_path: Path) -> str:
    """Return actual OS file creation/metadata date in YYYY-MM-DD format."""
    stat = Path(file_path).stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

def extract_pdf(file_path):
    """Extract text and file metadata from every page of a PDF file."""
    file_path = Path(file_path)
    created_date = get_file_created_date(file_path)
    extracted_pages = []
    pdf_document = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf_document, start=1):
        text = page.get_text("text").strip()
        if text:
            extracted_pages.append({
                "source": file_path.name,
                "page": page_number,
                "doc_type": "PDF",
                "created_date": created_date,
                "text": text
            })

    pdf_document.close()
    return extracted_pages

def extract_docx(file_path):
    """Extract text and file metadata from a Word document."""
    file_path = Path(file_path)
    created_date = get_file_created_date(file_path)
    document = Document(file_path)
    full_text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "source": file_path.name,
        "page": 1,
        "doc_type": "DOCX",
        "created_date": created_date,
        "text": full_text
    }]


def extract_txt(file_path):
    """Extract text and file metadata from a plain text file."""
    file_path = Path(file_path)
    created_date = get_file_created_date(file_path)
    text = file_path.read_text(encoding="utf-8").strip()

    return [{
        "source": file_path.name,
        "page": 1,
        "doc_type": "TXT",
        "created_date": created_date,
        "text": text
    }]


def extract_document(file_path):
    """Select the correct extraction method based on file extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)
    elif extension == ".docx":
        return extract_docx(file_path)
    elif extension == ".txt":
        return extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported document format: {extension}")


def extract_all_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """Extract text and metadata from all supported documents in a directory."""
    all_pages = []
    for file_path in sorted(Path(data_dir).iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            all_pages.extend(extract_document(file_path))
    return all_pages


def fixed_token_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into fixed token windows with overlap using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
        if start + chunk_size >= len(tokens):
            break

    return chunks

def sentence_chunking(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Group N sentences per chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]

def recursive_chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """Split text recursively preserving paragraph and sentence boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def detect_section_heading(text: str) -> str:
    """Extract section heading line if present."""
    for line in text.splitlines():
        clean = line.strip()
        if " - " in clean and len(clean) <= 80:
            return clean
    return "General"

def compute_hash(text: str) -> str:
    """Generate SHA-256 hash of normalized chunk text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def generate_stable_id(source: str, page: int, content_hash: str) -> str:
    """Generate shift-resistant deterministic UUID5 from source, page, and content hash."""
    unique_key = f"{source}::page_{page}::hash_{content_hash}"
    return str(uuid.uuid5(NAMESPACE_UUID, unique_key))

def build_chunks(docs: List[Dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Create chunks with actual file metadata and shift-resistant stable UUID5 IDs."""
    chunks = []
    for doc in docs:
        raw_chunks = recursive_chunking(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        current_section = detect_section_heading(doc["text"])
        doc_created_date = doc.get("created_date") or datetime.now().strftime("%Y-%m-%d")

        for idx, chunk_text in enumerate(raw_chunks):
            detected = detect_section_heading(chunk_text)
            if detected != "General":
                current_section = detected

            content_hash = compute_hash(chunk_text)
            stable_id = generate_stable_id(doc["source"], doc["page"], content_hash)

            chunks.append({
                "id": stable_id,
                "content_hash": content_hash,
                "text": chunk_text,
                "metadata": {
                    "source": doc["source"],
                    "page": doc["page"],
                    "section_heading": current_section,
                    "created_date": doc_created_date,
                    "chunk_index": idx,
                    "doc_type": doc.get("doc_type", "UNKNOWN"),
                    "content_hash": content_hash
                }
            })
    return chunks

def describe_chunk(chunk: Dict[str, Any]) -> str:
    """Return a formatted citation string for a chunk."""
    m = chunk["metadata"]
    return (
        f"[{m['source']} ({m['doc_type']}) | Page {m['page']} | "
        f"Section: {m['section_heading']} | Chunk #{m['chunk_index']} | Date: {m['created_date']}]"
    )