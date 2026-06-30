from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import (
    CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_FILE,
    EMBEDDING_MODEL,
    RANDOM_SEED,
)
from utils import cosine_sim_matrix, get_logger

logger = get_logger(__name__)

# Module-level model singleton (loaded lazily)
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model (CPU only)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _model.eval()
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of texts and return an (N, D) float32 array.

    Args:
        texts: Strings to embed.
        batch_size: Sentences per batch.
        show_progress: Show tqdm.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalized → dot == cosine
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_jd(jd_summary: str) -> np.ndarray:
    """
    Embed the JD full_summary and return a (1, D) array.
    """
    logger.info("Embedding JD summary")
    emb = embed_texts([jd_summary], show_progress=False)
    return emb  # shape (1, D)


def compute_candidate_embeddings(
    texts: list[str],
    candidate_ids: list[str],
    cache_path: Path = EMBEDDING_CACHE_FILE,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute or load cached candidate embeddings.

    Args:
        texts: Candidate full_text strings (same order as candidate_ids).
        candidate_ids: Candidate IDs (used to validate cache consistency).
        cache_path: Where to read/write the NPZ cache.
        force_recompute: Skip cache even if it exists.

    Returns:
        numpy array of shape (N, D), float32, L2-normalized.
    """
    cache_path = Path(cache_path)
    cache_ids_path = cache_path.with_suffix(".ids.npy")

    # ── Try cache ────────────────────────────────────────────────────────────
    if not force_recompute and cache_path.exists() and cache_ids_path.exists():
        logger.info("Loading candidate embeddings from cache: %s", cache_path)
        cached_ids = np.load(str(cache_ids_path), allow_pickle=True).tolist()
        if cached_ids == candidate_ids:
            data = np.load(str(cache_path))
            embs = data["embeddings"].astype(np.float32)
            logger.info("Cache hit — %d embeddings loaded", len(embs))
            return embs
        else:
            logger.warning("Cache ID mismatch — recomputing embeddings")

    # ── Compute ──────────────────────────────────────────────────────────────
    logger.info("Computing embeddings for %d candidates", len(texts))
    model = _get_model()

    all_embs: list[np.ndarray] = []
    bar = tqdm(total=len(texts), desc="Embedding candidates", unit=" cands")
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_emb = model.encode(
            batch,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embs.append(batch_emb.astype(np.float32))
        bar.update(len(batch))
    bar.close()

    embeddings = np.vstack(all_embs)  # (N, D)

    # ── Save to cache ─────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), embeddings=embeddings)
    np.save(str(cache_ids_path), np.array(candidate_ids, dtype=object))
    logger.info("Saved embeddings to %s", cache_path)

    return embeddings


def compute_semantic_scores(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity of each candidate vs. the JD embedding.

    Args:
        jd_embedding:         shape (1, D) — already L2-normalized
        candidate_embeddings: shape (N, D) — already L2-normalized

    Returns:
        1-D array of shape (N,) with scores in [0, 1].
    """
    # Since both sides are L2-normalized, dot product == cosine similarity
    scores = (jd_embedding @ candidate_embeddings.T).squeeze(0)   # (N,)
    # Map from [-1, 1] to [0, 1]
    scores = (scores + 1.0) / 2.0
    return scores.astype(np.float32)