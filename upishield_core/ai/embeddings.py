"""
Embedding utilities for fraud knowledge base.

Provides text embedding generation using sentence-transformers
with support for multiple models and batch processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    device: Optional[str] = None  # "cuda", "cpu", or None for auto


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence transformer embedding provider."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using random embeddings")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_batch(texts)

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        if self._model is None:
            # Fallback: random embeddings for testing
            logger.debug("Using random embeddings (model not loaded)")
            return np.random.rand(len(texts), self.dimension).astype(np.float32)

        batch_size = batch_size or self.config.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = self._model.encode(
                    batch,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=self.config.normalize,
                )
                all_embeddings.append(embeddings)
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                # Fallback for failed batch
                fallback = np.random.rand(len(batch), self.dimension).astype(np.float32)
                all_embeddings.append(fallback)

        if all_embeddings:
            result = np.vstack(all_embeddings)
        else:
            result = np.array([]).reshape(0, self.dimension)

        return result.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimension for the loaded model."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        # Default for all-MiniLM-L6-v2
        return 384


class HFInferenceProvider(EmbeddingProvider):
    """Hugging Face Inference API embedding provider."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self._dimension = 384

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_batch(texts)

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings via HF Inference API."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            import httpx

            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                )
                response.raise_for_status()
                embeddings = response.json()

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"HF Inference API failed: {e}")
            return np.random.rand(len(texts), self.dimension).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dimension = 768  # nomic-embed-text dimension

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_batch(texts)

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings via Ollama API."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            import httpx

            embeddings = []
            for text in texts:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": text},
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data.get("embedding", []))

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            return np.random.rand(len(texts), self.dimension).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = 1536  # text-embedding-3-small dimension

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_batch(texts)

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings via OpenAI API."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            embeddings = [d.embedding for d in response.data]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return np.random.rand(len(texts), self._dimension).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


# Provider factory
def create_embedding_provider(
    provider: str = "sentence_transformer",
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to create embedding provider."""
    providers = {
        "sentence_transformer": SentenceTransformerProvider,
        "hf_inference": HFInferenceProvider,
        "ollama": OllamaEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(providers.keys())}")

    return providers[provider](**kwargs)


# Async wrapper for batch embedding
async def embed_texts_async(
    provider: EmbeddingProvider,
    texts: List[str],
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Async wrapper for batch embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: provider.embed_batch(texts, batch_size),
    )


# Similarity utilities
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)

    return float(dot_product / (norm_a * norm_b + 1e-8))


def batch_cosine_similarity(
    query: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and all corpus vectors."""
    if query.ndim == 1:
        query = query.reshape(1, -1)

    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    corpus_norm = np.linalg.norm(corpus, axis=1, keepdims=True)

    similarity = np.dot(query, corpus.T) / (query_norm * corpus_norm.T + 1e-8)
    return similarity.flatten()


def get_top_k_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 5,
) -> List[tuple]:
    """Get top-k most similar embeddings."""
    similarities = batch_cosine_similarity(query_embedding, corpus_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]