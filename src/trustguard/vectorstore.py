"""Embedding and vector storage backed by SentenceTransformers and ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from trustguard.etl.chunker import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Anything that turns text into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding model; no document text leaves the machine."""

    def __init__(self, model_name: str) -> None:
        """Load the model lazily on first use to keep imports cheap.

        Args:
            model_name: Hugging Face model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as L2-normalized vectors (cosine similarity == dot product)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Serialize for logs and API responses."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4),
            "text": self.text,
        }


class PolicyVectorStore:
    """Thin wrapper over a ChromaDB collection configured for cosine similarity."""

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path | None = None,
        collection_name: str = "policies",
    ) -> None:
        """Open (or create) the collection.

        Args:
            embedder: Embedding implementation.
            persist_dir: Directory for on-disk storage. ``None`` keeps everything in memory.
            collection_name: Name of the Chroma collection.
        """
        self.embedder = embedder
        self.client: ClientAPI = (
            chromadb.PersistentClient(path=str(persist_dir))
            if persist_dir is not None
            else chromadb.EphemeralClient()
        )
        self.collection_name = collection_name
        self.collection: Collection = self._open_collection()

    def _open_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        """Embed and upsert chunks. Returns the number written."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed([c.embedding_text for c in batch]),  # type: ignore[arg-type]
                metadatas=[c.to_metadata() for c in batch],  # type: ignore[arg-type]
            )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from the given document."""
        self.collection.delete(where={"source": source})

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, best first.

        Scores are cosine similarities in ``[-1, 1]`` (higher is more similar).
        """
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),  # type: ignore[arg-type]
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = (result["documents"] or [[]])[0]
        metas = (result["metadatas"] or [[]])[0]
        dists = (result["distances"] or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source=str(meta["source"]),
                page=int(cast(int, meta["page"])),
                score=1.0 - float(dist),
            )
            for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True)
        ]

    def count(self) -> int:
        """Number of stored chunks."""
        return self.collection.count()

    def sources(self) -> list[str]:
        """Distinct document names currently indexed."""
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source"]) for m in metas})

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._open_collection()
