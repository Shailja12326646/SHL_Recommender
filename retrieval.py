"""
retrieval.py — Agent 1: Retrieval Engine

Hybrid BM25 + dense embedding retrieval with metadata filtering.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from catalog import SHLAssessment, SENIORITY_ALIASES

# ─────────────────────────────────────────
# Optional heavy imports (graceful fallback)
# ─────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("rank-bm25 not available, falling back to TF-IDF")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False
    print("sentence-transformers/faiss not available, using BM25 only")


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_CACHE_PATH = Path(os.getenv("EMBED_CACHE_PATH", "cache/embeddings.pkl"))


class RetrievalEngine:
    def __init__(self, assessments: List[SHLAssessment]):
        self.assessments = assessments
        self._valid_urls = {a.url for a in assessments}

        # BM25
        if HAS_BM25:
            tokenized = [a.searchable_text.lower().split() for a in assessments]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

        # Dense embeddings
        self.embedder = None
        self.index = None
        self.embeddings = None
        if HAS_DENSE:
            self._init_dense()

    def _init_dense(self):
        """Build or load cached embeddings."""
        if EMBED_CACHE_PATH.exists():
            try:
                with open(EMBED_CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                    if cache.get("count") == len(self.assessments):
                        self.embeddings = cache["embeddings"]
                        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
                        self._build_faiss_index()
                        print(f"[Retrieval] Loaded {len(self.assessments)} cached embeddings")
                        return
            except Exception as e:
                print(f"[Retrieval] Cache load failed: {e}")

        print(f"[Retrieval] Building embeddings for {len(self.assessments)} assessments...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        texts = [a.embedding_text for a in self.assessments]
        self.embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        # Cache
        try:
            with open(EMBED_CACHE_PATH, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "count": len(self.assessments)}, f)
        except Exception:
            pass

        self._build_faiss_index()
        print("[Retrieval] Embeddings ready")

    def _build_faiss_index(self):
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        self.index.add(self.embeddings.astype(np.float32))

    def is_valid_url(self, url: str) -> bool:
        return url in self._valid_urls

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        seniority_buckets: Optional[List[str]] = None,
        require_personality: bool = False,
        require_cognitive: bool = False,
        require_simulation: bool = False,
        max_duration: Optional[int] = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
        language: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
        excluded_ids: Optional[List[str]] = None,
    ) -> List[Tuple[SHLAssessment, float]]:
        """
        Hybrid retrieval: BM25 recall → embedding rerank → metadata filter.
        Returns list of (assessment, score) sorted by score descending.
        """
        query_lower = query.lower()

        # ── Step 1: BM25 recall (top 80) ──────────────────────────────
        bm25_scores = {}
        if self.bm25 is not None:
            tokens = query_lower.split()
            scores = self.bm25.get_scores(tokens)
            # top 80 candidates
            top_indices = np.argsort(scores)[::-1][:80]
            max_bm25 = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0
            for idx in top_indices:
                bm25_scores[idx] = float(scores[idx]) / max_bm25
        else:
            # fallback: simple word overlap
            tokens = set(query_lower.split())
            for i, a in enumerate(self.assessments):
                a_tokens = set(a.searchable_text.lower().split())
                overlap = len(tokens & a_tokens)
                bm25_scores[i] = float(overlap) / (len(tokens) + 1e-9)

        # ── Step 2: Dense embedding similarity ────────────────────────
        dense_scores = {}
        if self.embedder is not None and self.index is not None:
            q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
            dist, indices = self.index.search(q_emb, min(80, len(self.assessments)))
            for dist_val, idx in zip(dist[0], indices[0]):
                dense_scores[int(idx)] = float(dist_val)  # already cosine similarity

        # ── Step 3: Combine scores ─────────────────────────────────────
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())
        combined = {}
        for idx in all_indices:
            b = bm25_scores.get(idx, 0.0)
            d = dense_scores.get(idx, 0.0)
            # Weighted: semantic 0.5, BM25 0.3, small baseline 0.2
            combined[idx] = d * 0.55 + b * 0.35 + 0.1
        
        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # ── Step 4: Metadata filtering ────────────────────────────────
        results = []
        excluded_ids = excluded_ids or []
        
        for idx, score in ranked:
            a = self.assessments[idx]

            # Exclude specific IDs
            if a.id in excluded_ids:
                continue

            # Seniority filter
            if seniority_buckets:
                if a.job_level_buckets:  # only filter if assessment has level data
                    overlap = set(a.job_level_buckets) & set(seniority_buckets)
                    if not overlap:
                        continue

            # Category filters
            if require_personality and "Personality & Behavior" not in a.categories:
                continue
            if require_cognitive and "Ability & Aptitude" not in a.categories:
                continue
            if require_simulation and "Simulations" not in a.categories:
                continue
            if category_filter:
                if not any(cat in a.categories for cat in category_filter):
                    continue

            # Duration filter
            if max_duration is not None and a.duration_minutes is not None:
                if a.duration_minutes > max_duration:
                    continue

            # Remote filter
            if remote_only and not a.remote_support:
                continue
            if adaptive_only and not a.adaptive_support:
                continue

            # Language filter
            if language:
                lang_lower = language.lower()
                has_lang = any(lang_lower in l.lower() for l in a.languages) or len(a.languages) == 0
                if not has_lang:
                    continue

            results.append((a, score))
            if len(results) >= top_k:
                break

        return results

    def get_by_url(self, url: str) -> Optional[SHLAssessment]:
        for a in self.assessments:
            if a.url == url:
                return a
        return None

    def get_by_name(self, name: str) -> Optional[SHLAssessment]:
        name_lower = name.lower().strip()
        for a in self.assessments:
            if a.name.lower() == name_lower:
                return a
        # fuzzy substring
        for a in self.assessments:
            if name_lower in a.name.lower() or a.name.lower() in name_lower:
                return a
        return None

    def get_by_names(self, names: List[str]) -> List[SHLAssessment]:
        results = []
        for name in names:
            a = self.get_by_name(name)
            if a:
                results.append(a)
        return results
