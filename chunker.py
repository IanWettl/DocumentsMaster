"""
Step 1: Walk a folder, extract text from files, split into overlapping chunks.
"""
import os

# --- config ---
SCOPE_DIR = os.path.expanduser("~/Documents")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".DS_Store"}
SKIP_EXTENSIONS = {".pem", ".key", ".env"}
ALLOWED_EXTENSIONS = {".txt", ".md", ".py", ".cpp", ".h", ".csv", ".json"}
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def extract_text(filepath):
    """Read a file's text content. Returns None if unreadable/unsupported."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in SKIP_EXTENSIONS:
        return None

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return None

    if ext == ".docx":
        try:
            import docx
            d = docx.Document(filepath)
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            return None

    if ext in ALLOWED_EXTENSIONS:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None

    return None  # unsupported type, skip


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping fixed-size chunks."""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but re-cover the overlap

    return chunks


def walk_and_chunk(scope_dir=SCOPE_DIR):
    """Walk the directory tree, yield (filepath, chunk_index, chunk_text, mtime)."""
    for root, dirs, files in os.walk(scope_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]  # prune in-place

        for fname in files:
            filepath = os.path.join(root, fname)
            text = extract_text(filepath)
            if not text:
                continue

            mtime = os.path.getmtime(filepath)
            for i, chunk in enumerate(chunk_text(text)):
                yield {
                    "filepath": filepath,
                    "chunk_index": i,
                    "text": chunk,
                    "mtime": mtime,
                }


if __name__ == "__main__":
    # quick sanity check — just print what it would index, don't embed yet
    count = 0
    for item in walk_and_chunk():
        count += 1
        if count <= 3:
            print(f"--- {item['filepath']} [chunk {item['chunk_index']}] ---")
            print(item["text"][:200], "...\n")
    print(f"Total chunks found: {count}")