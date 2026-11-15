import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: dict


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    
    if suffix in [".txt", ".md", ".markdown", ".rst"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            return f"[PDF файл: {file_path.name} - pdfplumber не установлен]"
    else:
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return f"[Неподдерживаемый файл: {file_path.name}]"


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def load_knowledge_base(data_dir: Path, chunk_size: int = 512, overlap: int = 100) -> List[DocumentChunk]:
    kb_zip = data_dir / "knowledge-base.zip"
    if not kb_zip.exists():
        raise FileNotFoundError(f"knowledge-base.zip не найден в {data_dir}")
    
    extract_dir = data_dir / "kb_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(kb_zip, "r") as zf:
        zf.extractall(extract_dir)
    
    chunks = []
    chunk_counter = 0
    
    for file_path in sorted(extract_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        
        text = extract_text_from_file(file_path)
        if not text.strip():
            continue
        
        file_chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk_text_content in enumerate(file_chunks):
            chunk_id = f"doc_{chunk_counter}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=chunk_text_content,
                metadata={
                    "source": str(file_path.relative_to(extract_dir)),
                    "chunk_index": i,
                    "total_chunks": len(file_chunks),
                }
            ))
            chunk_counter += 1
    
    return chunks