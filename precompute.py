import os
import pickle
from catalog import load_catalog
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_CACHE_PATH = Path("cache/embeddings.pkl")

def precompute():
    print("Loading catalog...")
    assessments = load_catalog("catalog.txt")
    print(f"Loaded {len(assessments)} assessments")
    
    os.makedirs("cache", exist_ok=True)
    
    print(f"Downloading/loading model {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    
    print("Generating embeddings...")
    texts = [a.embedding_text for a in assessments]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    with open(EMBED_CACHE_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "count": len(assessments)}, f)
    
    print(f"Saved {len(assessments)} embeddings to {EMBED_CACHE_PATH}")

if __name__ == "__main__":
    precompute()
