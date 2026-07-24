"""
Step 2: Embed each chunk and store it (text + vector + metadata) in SQLite.
"""
import sqlite3
import json
from sentence_transformers import SentenceTransformer
from chunker import walk_and_chunk

DB_PATH = "rag_index.db"

# small, fast, runs locally, no API key needed
# (384-dimension embeddings — good enough to start with)
MODEL_NAME = "all-MiniLM-L6-v2"


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            mtime REAL NOT NULL,
            embedding TEXT NOT NULL,   -- JSON-encoded list of floats
            UNIQUE(filepath, chunk_index)
        )
    """)
    conn.commit()


def already_indexed(conn, filepath, mtime):
    """Skip re-embedding a file if it hasn't changed since last run."""
    row = conn.execute(
        "SELECT mtime FROM chunks WHERE filepath = ? LIMIT 1", (filepath,)
    ).fetchone()
    return row is not None and row[0] == mtime


def build_index(scope_dir, db_path=DB_PATH):
    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    conn = sqlite3.connect(db_path)
    init_db(conn)

    seen_files = set()
    new_chunks = 0
    skipped_files = 0

    # group chunks by file so we can check the "already indexed" shortcut per file
    by_file = {}
    for item in walk_and_chunk(scope_dir):
        by_file.setdefault(item["filepath"], []).append(item)

    for filepath, chunks in by_file.items():
        mtime = chunks[0]["mtime"]

        if already_indexed(conn, filepath, mtime):
            skipped_files += 1
            continue

        # file changed or is new — clear old chunks for it, then re-embed
        conn.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))

        texts = [c["text"] for c in chunks]
        vectors = model.encode(texts, show_progress_bar=False)

        for c, vec in zip(chunks, vectors):
            conn.execute(
                "INSERT INTO chunks (filepath, chunk_index, text, mtime, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (c["filepath"], c["chunk_index"], c["text"], c["mtime"], json.dumps(vec.tolist())),
            )
            new_chunks += 1

        seen_files.add(filepath)

    conn.commit()
    conn.close()
    print(f"Indexed {new_chunks} new chunks from {len(seen_files)} files.")
    print(f"Skipped {skipped_files} unchanged files.")


if __name__ == "__main__":
    import sys
    scope = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ianwe\OneDrive\Documents"
    import os
    build_index(os.path.expanduser(scope))