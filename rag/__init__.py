"""
RAG (Retrieval-Augmented Generation) Module for Concordia Pipeline v3

This module provides the knowledge base infrastructure for dynamic
specification retrieval, replacing hardcoded rules from v2.
"""

from .embeddings import EmbeddingProvider, LocalEmbeddings, VoyageEmbeddings, MockEmbeddings, get_embedding_provider
from .vector_store import VectorStore
from .retriever import SpecificationRetriever
from .indexer import DocumentIndexer

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddings",
    "VoyageEmbeddings",
    "MockEmbeddings",
    "get_embedding_provider",
    "VectorStore",
    "SpecificationRetriever",
    "DocumentIndexer",
]
