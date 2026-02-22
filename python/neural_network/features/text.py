"""Text embedding pipeline using sentence-transformers.

Encodes free-text proposal descriptions into dense vector representations.
Embeddings are cached to a memory-mapped binary file on disk keyed by a
truncated SHA-256 hash of the input text so repeated runs skip the
transformer forward pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from neural_network.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.json"
_EMBEDDINGS_FILE = "embeddings.dat"


class TextEmbedder:
    """Encode proposal text into dense vectors with on-disk caching.

    Cache layout inside *cache_dir*::

        index.json      — ``{"dim": 384, "count": N, "entries": {"<hash>": row_idx, …}}``
        embeddings.dat  — raw ``float32`` binary, shape ``(count, dim)``

    The binary file is read via ``numpy.memmap`` so only the pages
    actually accessed are loaded into RAM.

    Parameters:
        model_name: HuggingFace model ID (default from settings).
        cache_dir:  Directory for cache files (created automatically).
        settings:   Application settings (injected for testability).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.text_encoder_model
        self._cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self._model = None
        self._device: str = "cpu"
        self._index: dict | None = None

    # ── model lifecycle ───────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the sentence-transformer, selecting GPU when available."""
        from sentence_transformers import SentenceTransformer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("Loaded %s on %s", self._model_name, self._device)

    def _ensure_model(self) -> None:
        if self._model is None:
            self._load_model()

    # ── cache management ──────────────────────────────────────────────

    def _load_index(self) -> None:
        path = self._cache_dir / _INDEX_FILE
        if path.exists():
            self._index = json.loads(path.read_text())
        else:
            self._index = {"dim": 0, "count": 0, "entries": {}}

    def _save_index(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / _INDEX_FILE).write_text(json.dumps(self._index))

    def _read_cached(self) -> Optional[np.ndarray]:
        """Memory-map the existing embedding store (read-only)."""
        if self._index is None:
            self._load_index()
        assert self._index is not None
        if self._index["count"] == 0 or self._index["dim"] == 0:
            return None
        return np.memmap(
            self._cache_dir / _EMBEDDINGS_FILE,
            dtype=np.float32,
            mode="r",
            shape=(self._index["count"], self._index["dim"]),
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        """Truncated SHA-256 used as cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def clear_cache(self) -> None:
        """Delete all cached embeddings."""
        for name in (_INDEX_FILE, _EMBEDDINGS_FILE):
            p = self._cache_dir / name
            if p.exists():
                p.unlink()
        self._index = None

    # ── public API ────────────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts, hitting the disk cache first.

        Args:
            texts: Raw text strings.

        Returns:
            Array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            self._ensure_model()
            assert self._model is not None
            dim = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        self._ensure_model()
        if self._index is None:
            self._load_index()
        assert self._index is not None and self._model is not None

        dim: int = self._model.get_sentence_embedding_dimension()
        if self._index["dim"] == 0:
            self._index["dim"] = dim

        n = len(texts)
        result = np.empty((n, dim), dtype=np.float32)
        hashes = [self._hash_text(t) for t in texts]
        cached = self._read_cached()

        uncached_idx: list[int] = []
        uncached_texts: list[str] = []
        for i, h in enumerate(hashes):
            if h in self._index["entries"] and cached is not None:
                result[i] = cached[self._index["entries"][h]]
            else:
                uncached_idx.append(i)
                uncached_texts.append(texts[i])

        if not uncached_texts:
            return result

        logger.info(
            "Encoding %d new texts (%d cached)",
            len(uncached_texts),
            n - len(uncached_texts),
        )
        new_vecs: np.ndarray = self._model.encode(
            uncached_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Append to the binary cache file
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_dir / _EMBEDDINGS_FILE, "ab") as f:
            for j, idx in enumerate(uncached_idx):
                vec = new_vecs[j]
                result[idx] = vec
                f.write(vec.tobytes())
                self._index["entries"][hashes[idx]] = self._index["count"]
                self._index["count"] += 1

        self._save_index()
        return result

    def embed_single(self, text: str) -> np.ndarray:
        """Encode one text string.

        Args:
            text: Raw text.

        Returns:
            Array of shape ``(embedding_dim,)``.
        """
        return self.embed_batch([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality (loads the model if needed)."""
        self._ensure_model()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()
