"""
Vector Store Interface using ChromaDB

Provides persistent storage and retrieval of specification embeddings.
ChromaDB runs embedded (no external infrastructure needed).

Usage:
    from rag.embeddings import LocalEmbeddings
    from rag.vector_store import VectorStore

    # Initialize
    embeddings = LocalEmbeddings()
    store = VectorStore(embeddings, persist_dir="./chroma_db")

    # Add documents
    store.add_documents(
        ids=["rule_1", "rule_2"],
        documents=["SEX must be Male, Female, or Unknown", "TRIAL must match NCT format"],
        metadatas=[{"variable": "SEX", "type": "validation"}, {"variable": "TRIAL", "type": "validation"}]
    )

    # Query
    results = store.query("What are valid values for SEX?", n_results=3)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        """Iterate over results as tuples of (id, document, metadata, distance)."""
        for i in range(len(self.ids)):
            yield (
                self.ids[i],
                self.documents[i],
                self.metadatas[i] if self.metadatas else {},
                self.distances[i] if self.distances else 0.0
            )

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for easy consumption."""
        return [
            {
                "id": self.ids[i],
                "document": self.documents[i],
                "metadata": self.metadatas[i] if self.metadatas else {},
                "distance": self.distances[i] if self.distances else 0.0,
                "score": 1 - self.distances[i] if self.distances else 1.0  # Convert distance to similarity
            }
            for i in range(len(self.ids))
        ]


class VectorStore:
    """
    ChromaDB-based vector store for specification documents.

    Features:
        - Persistent storage (survives restarts)
        - Automatic embedding generation
        - Metadata filtering
        - Configurable similarity threshold
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str = "./chroma_db",
        collection_name: str = "specifications"
    ):
        """
        Initialize the vector store.

        Args:
            embedding_provider: EmbeddingProvider instance for generating vectors
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_provider = embedding_provider
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _initialize(self):
        """Lazy initialization of ChromaDB client and collection."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb not installed. "
                    "Run: pip install chromadb"
                )

            logger.info(f"Initializing ChromaDB at: {self.persist_dir}")

            # Create persist directory if needed
            os.makedirs(self.persist_dir, exist_ok=True)

            # Initialize persistent client
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "embedding_model": self.embedding_provider.model_name,
                    "embedding_dimension": self.embedding_provider.dimension,
                    "description": "Concordia Pipeline v3 Specifications"
                }
            )

            logger.info(
                f"Collection '{self.collection_name}' initialized. "
                f"Document count: {self._collection.count()}"
            )

    @property
    def collection(self):
        """Get the ChromaDB collection, initializing if needed."""
        self._initialize()
        return self._collection

    @property
    def client(self):
        """Get the ChromaDB client, initializing if needed."""
        self._initialize()
        return self._client

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            ids: Unique identifiers for each document
            documents: Text content to embed and store
            metadatas: Optional metadata dicts for each document

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        if len(ids) != len(documents):
            raise ValueError("ids and documents must have same length")

        if metadatas and len(metadatas) != len(documents):
            raise ValueError("metadatas must have same length as documents")

        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_provider.embed(documents)

        logger.info(f"Adding {len(documents)} documents to collection...")
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents. Total count: {self.collection.count()}")
        return len(documents)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> QueryResult:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Text to search for
            n_results: Maximum number of results to return
            where: Metadata filter (e.g., {"variable": "SEX"})
            where_document: Document content filter
            include_distances: Whether to include distance scores

        Returns:
            QueryResult with matching documents
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_single(query_text)

        # Build include list
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")

        # ChromaDB requires $and operator for multiple conditions
        if where and len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}

        # Execute query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )

        # Flatten results (ChromaDB returns nested lists)
        return QueryResult(
            ids=results["ids"][0] if results["ids"] else [],
            documents=results["documents"][0] if results["documents"] else [],
            metadatas=results["metadatas"][0] if results.get("metadatas") else [],
            distances=results["distances"][0] if results.get("distances") else []
        )

    def query_by_metadata(
        self,
        where: Dict[str, Any],
        n_results: int = 100
    ) -> QueryResult:
        """
        Retrieve documents by metadata without semantic search.

        Args:
            where: Metadata filter (can have multiple conditions)
            n_results: Maximum results

        Returns:
            QueryResult with matching documents
        """
        # ChromaDB requires $and operator for multiple conditions
        if len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}

        results = self.collection.get(
            where=where,
            limit=n_results,
            include=["documents", "metadatas"]
        )

        return QueryResult(
            ids=results["ids"] if results["ids"] else [],
            documents=results["documents"] if results["documents"] else [],
            metadatas=results["metadatas"] if results.get("metadatas") else [],
            distances=[]  # No distances for metadata-only queries
        )

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Dict with document and metadata, or None if not found
        """
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )

        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results.get("metadatas") else {}
            }
        return None

    def delete(self, ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
        return len(ids)

    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self._collection = None
        # Recreate the collection immediately
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": self.embedding_provider.model_name,
                "embedding_dimension": self.embedding_provider.dimension,
                "description": "Concordia Pipeline v3 Specifications"
            }
        )
        logger.info("Collection cleared")

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_provider.model_name,
            "embedding_dimension": self.embedding_provider.dimension
        }
