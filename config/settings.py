"""
Configuration Settings for Concordia Pipeline v3

Centralizes all configuration including:
    - Embedding provider selection
    - API keys
    - File paths
    - RAG parameters

Usage:
    from config import get_settings

    settings = get_settings()
    print(settings.embedding_provider)  # "local" or "voyage"
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings with defaults."""

    # === Embedding Configuration ===
    # Provider: "local" (sentence-transformers) or "voyage" (Voyage AI)
    embedding_provider: str = "local"

    # Local embedding model (when provider="local")
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Voyage AI settings (when provider="voyage")
    voyage_api_key: Optional[str] = None
    voyage_model: str = "voyage-3"

    # HuggingFace settings (for local embeddings)
    huggingface_token: Optional[str] = None

    # === LLM Configuration ===
    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"  # Default to Sonnet for cost efficiency
    llm_orchestrator_model: str = "claude-opus-4-5-20251101"  # Opus for orchestration

    # === Paths ===
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    knowledge_base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "knowledge_base")
    chroma_persist_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "chroma_db")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output")

    # === RAG Parameters ===
    chroma_collection_name: str = "dm_specifications"
    default_retrieval_results: int = 5
    max_context_tokens: int = 4000

    # === Processing Parameters ===
    agent_timeout_seconds: int = 120
    max_retries: int = 1
    batch_size: int = 100

    # === Feature Flags ===
    use_llm_fallback: bool = True
    enable_review_agent: bool = True
    verbose_logging: bool = False

    def __post_init__(self):
        """Initialize paths and load environment variables."""
        # Convert string paths to Path objects if needed
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if isinstance(self.knowledge_base_dir, str):
            self.knowledge_base_dir = Path(self.knowledge_base_dir)
        if isinstance(self.chroma_persist_dir, str):
            self.chroma_persist_dir = Path(self.chroma_persist_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Load API keys from environment if not set
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.voyage_api_key:
            self.voyage_api_key = os.getenv("VOYAGE_API_KEY")

        if not self.huggingface_token:
            self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # Auto-switch to voyage if API key is present and local not explicitly set
        if self.voyage_api_key and self.embedding_provider == "local":
            # Only switch if EMBEDDING_PROVIDER env var is not explicitly set to local
            env_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
            if env_provider == "voyage":
                self.embedding_provider = "voyage"

        # Create directories
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate settings and return True if valid."""
        errors = []

        # Check embedding provider configuration
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            errors.append("Voyage API key required when embedding_provider='voyage'")

        # Check LLM configuration if LLM features enabled
        if self.use_llm_fallback and not self.anthropic_api_key:
            logger.warning("Anthropic API key not set - LLM fallback will be disabled")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        return True

    def get_spec_path(self, spec_name: str = "DM_Harmonization_Spec_v1.4.docx") -> Path:
        """Get path to a specification document."""
        return self.knowledge_base_dir / "specs" / spec_name

    def get_rules_path(self, rules_name: str) -> Path:
        """Get path to a rules file."""
        return self.knowledge_base_dir / "rules" / rules_name

    def get_codelist_path(self, codelist_name: str) -> Path:
        """Get path to a codelist file."""
        return self.knowledge_base_dir / "codelists" / codelist_name


# Singleton instance
_settings: Optional[Settings] = None


def get_settings(**overrides) -> Settings:
    """
    Get the settings instance, creating if needed.

    Args:
        **overrides: Override any default settings

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None or overrides:
        # Load from environment
        env_settings = {
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "voyage_api_key": os.getenv("VOYAGE_API_KEY"),
            "verbose_logging": os.getenv("VERBOSE_LOGGING", "").lower() == "true",
        }

        # Remove None values
        env_settings = {k: v for k, v in env_settings.items() if v is not None}

        # Merge with overrides
        env_settings.update(overrides)

        _settings = Settings(**env_settings)

    return _settings


def reset_settings():
    """Reset the settings singleton (useful for testing)."""
    global _settings
    _settings = None
