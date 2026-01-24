"""
Embedding Provider Interface

Provides a unified interface for generating embeddings, allowing easy switching
between local (sentence-transformers) and cloud (Voyage AI) providers.

Usage:
    # Local embeddings (default for development)
    provider = LocalEmbeddings()

    # Voyage AI (for production)
    provider = VoyageEmbeddings(api_key="your-key")

    # Mock embeddings (for testing without network access)
    provider = MockEmbeddings()

    # Generate embeddings
    vectors = provider.embed(["text1", "text2"])
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for logging/tracking."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        pass

    def embed_single(self, text: str) -> List[float]:
        """Convenience method for embedding a single text."""
        return self.embed([text])[0]


class MockEmbeddings(EmbeddingProvider):
    """
    Mock embeddings for testing without network access.

    Generates deterministic pseudo-embeddings based on text hash.
    Similar texts will have somewhat similar embeddings due to
    shared n-gram hashes, but these are NOT semantically meaningful.

    Use only for testing infrastructure - NOT for actual retrieval quality.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize mock embedding provider.

        Args:
            dimension: Embedding dimension (default 384 to match MiniLM)
        """
        self._dimension = dimension
        logger.warning(
            "Using MockEmbeddings - these are NOT semantically meaningful! "
            "Use LocalEmbeddings or VoyageEmbeddings for production."
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "mock/deterministic-hash"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic pseudo-embeddings based on text hash."""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            # Use multiple hash variants for more dimensions
            embedding = []
            text_lower = text.lower()

            # Generate embedding dimensions from character n-grams
            for i in range(self._dimension):
                # Use different seeds for each dimension
                seed_text = f"{text_lower}_{i}"
                hash_val = int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16)
                # Normalize to [-1, 1] range
                val = (hash_val / 0xFFFFFFFF) * 2 - 1
                embedding.append(val)

            # Normalize to unit length
            norm = sum(v**2 for v in embedding) ** 0.5
            embedding = [v / norm for v in embedding]

            embeddings.append(embedding)

        return embeddings


class LocalEmbeddings(EmbeddingProvider):
    """
    Local embeddings using sentence-transformers.

    No API key required. Good for development and testing.
    Models are downloaded and cached locally on first use.

    Recommended models:
        - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
        - all-mpnet-base-v2: Better quality, slower (768 dimensions)
        - BAAI/bge-small-en-v1.5: Good balance (384 dimensions)
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", huggingface_token: Optional[str] = None):
        """
        Initialize local embedding provider.

        Args:
            model_name: HuggingFace model name for sentence-transformers
            huggingface_token: Optional HuggingFace token for authenticated access
        """
        self._model_name = model_name
        self._model = None  # Lazy loading
        self._dimension = self.MODEL_DIMENSIONS.get(model_name)
        self._hf_token = huggingface_token

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"Loading local embedding model: {self._model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                import os

                # Set HuggingFace token if provided
                if self._hf_token:
                    os.environ["HF_TOKEN"] = self._hf_token
                    os.environ["HUGGINGFACE_TOKEN"] = self._hf_token
                    logger.info("Using HuggingFace token for authentication")

                self._model = SentenceTransformer(self._model_name, token=self._hf_token)
                # Get actual dimension from model if not in our mapping
                if self._dimension is None:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384  # Default fallback

    @property
    def model_name(self) -> str:
        return f"local/{self._model_name}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        self._load_model()

        if not texts:
            return []

        # sentence-transformers returns numpy array
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )

        # Convert to list of lists for compatibility
        return embeddings.tolist()


class VoyageEmbeddings(EmbeddingProvider):
    """
    Cloud embeddings using Voyage AI.

    Higher quality embeddings, requires API key.
    Recommended for production deployments.

    Models:
        - voyage-3: Latest, best quality (1024 dimensions)
        - voyage-3-lite: Faster, smaller (512 dimensions)
        - voyage-code-3: Optimized for code (1024 dimensions)
    """

    MODEL_DIMENSIONS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-code-3": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        batch_size: int = 128
    ):
        """
        Initialize Voyage AI embedding provider.

        Args:
            api_key: Voyage AI API key
            model: Model name (voyage-3, voyage-3-lite, voyage-code-3)
            batch_size: Max texts per API call
        """
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client = None  # Lazy loading
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1024)

    def _load_client(self):
        """Lazy load the Voyage client on first use."""
        if self._client is None:
            logger.info(f"Initializing Voyage AI client with model: {self._model}")
            try:
                import voyageai
                self._client = voyageai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "voyageai not installed. "
                    "Run: pip install voyageai"
                )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"voyage/{self._model}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Voyage AI API."""
        self._load_client()

        if not texts:
            return []

        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            result = self._client.embed(
                batch,
                model=self._model,
                input_type="document"  # Optimize for retrieval
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings


def get_embedding_provider(
    provider_type: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    huggingface_token: Optional[str] = None,
    allow_mock_fallback: bool = True
) -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider.

    Args:
        provider_type: "local", "voyage", or "mock"
        api_key: Required for "voyage" provider
        model_name: Optional model override
        huggingface_token: Optional HuggingFace token for local provider
        allow_mock_fallback: If True, fall back to mock if local fails

    Returns:
        Configured EmbeddingProvider instance
    """
    if provider_type == "mock":
        return MockEmbeddings()

    if provider_type == "local":
        if allow_mock_fallback:
            try:
                provider = LocalEmbeddings(
                    model_name=model_name or "all-MiniLM-L6-v2",
                    huggingface_token=huggingface_token
                )
                # Try to load model to check availability
                _ = provider.dimension
                return provider
            except Exception as e:
                logger.warning(
                    f"LocalEmbeddings failed ({e}), falling back to MockEmbeddings. "
                    "This is OK for testing but not for production."
                )
                return MockEmbeddings()
        else:
            return LocalEmbeddings(
                model_name=model_name or "all-MiniLM-L6-v2",
                huggingface_token=huggingface_token
            )

    elif provider_type == "voyage":
        if not api_key:
            raise ValueError("API key required for Voyage embeddings")
        return VoyageEmbeddings(
            api_key=api_key,
            model=model_name or "voyage-3"
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
