"""
Step 3: Embed a query, retrieve the most similar chunks (brute-force cosine similarity).
"""
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from embed_store import DB_PATH, MODEL_NAME


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, top_k=5, db_path=DB_PATH):
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(query)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT filepath, chunk_index, text, embedding FROM chunks").fetchall()
    conn.close()

    scored = []
    for filepath, chunk_index, text, embedding_json in rows:
        vec = json.loads(embedding_json)
        score = cosine_similarity(query_vec, vec)
        scored.append((score, filepath, chunk_index, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "what did I write about?"
    results = search(query)

    print(f"\nQuery: {query}\n")
    for score, filepath, chunk_index, text in results:
        print(f"[{score:.3f}] {filepath} (chunk {chunk_index})")
        print(f"    {text[:150].strip()}...\n")