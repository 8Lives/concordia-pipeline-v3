"""
Pipeline Orchestrator - Coordinates Agent Execution

The Orchestrator manages the full harmonization pipeline:
1. Initializes RAG and LLM infrastructure
2. Runs agents in sequence (Ingest → Map → Harmonize → QC → Review)
3. Handles errors and provides recovery options
4. Tracks lineage and generates reports

Key Features:
- RAG-first architecture: specs loaded from vector store
- LLM-powered: Claude for value resolution and review
- Fallback capability: works without RAG/LLM using hardcoded rules
- Progress callbacks for UI integration
- Comprehensive logging and error handling
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

from agents.base import PipelineContext, AgentResult, AgentStatus
from agents.ingest_agent import IngestAgent
from agents.map_agent import MapAgent
from agents.harmonize_agent import HarmonizeAgent
from agents.qc_agent import QCAgent
from agents.review_agent import ReviewAgent
from rag.embeddings import get_embedding_provider, MockEmbeddings
from rag.vector_store import VectorStore
from rag.indexer import DocumentIndexer
from rag.retriever import SpecificationRetriever
from config.settings import get_settings, Settings

# Import LLM service (optional)
try:
    from llm.service import LLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMService = None

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a complete pipeline run."""
    success: bool
    harmonized_data: Optional[pd.DataFrame] = None
    qc_report: Optional[pd.DataFrame] = None
    review_result: Optional[Dict[str, Any]] = None
    mapping_log: Optional[List[Dict[str, Any]]] = None
    lineage: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stage_results: Dict[str, AgentResult] = field(default_factory=dict)
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type for progress callback: (stage_name, status, message, progress_pct)
ProgressCallback = Callable[[str, str, str, float], None]


class PipelineOrchestrator:
    """
    Orchestrates the harmonization pipeline.

    Usage:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(
            input_file="path/to/data.csv",
            trial_id="NCT12345678"
        )
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[ProgressCallback] = None,
        use_rag: bool = True,
        use_llm: bool = True,
        enable_review: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            settings: Configuration settings (uses defaults if not provided)
            progress_callback: Optional callback for progress updates
            use_rag: Whether to use RAG for spec retrieval (True) or fallback (False)
            use_llm: Whether to use LLM for value resolution and review
            enable_review: Whether to run the LLM review stage
        """
        self.settings = settings or get_settings()
        self.progress_callback = progress_callback
        self.use_rag = use_rag
        self.use_llm = use_llm
        self.enable_review = enable_review

        # RAG components (initialized lazily)
        self._retriever: Optional[SpecificationRetriever] = None
        self._vector_store: Optional[VectorStore] = None
        self._rag_initialized = False

        # LLM components (initialized lazily)
        self._llm_service = None
        self._llm_initialized = False

    def _initialize_llm(self) -> bool:
        """
        Initialize LLM service for value resolution and review.

        Returns:
            True if LLM initialized successfully, False otherwise
        """
        if self._llm_initialized:
            return self._llm_service is not None and self._llm_service.is_configured

        self._update_progress("init", "running", "Initializing LLM service...", 0.6)

        if not LLM_AVAILABLE:
            logger.warning("LLM module not available")
            self._llm_initialized = True
            return False

        try:
            # Get API key from settings or environment
            api_key = self.settings.anthropic_api_key

            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            self._llm_service = LLMService(api_key=api_key)
            self._llm_initialized = True

            if self._llm_service.is_configured:
                self._update_progress("init", "success", "LLM service initialized", 0.7)
                logger.info("LLM service initialized successfully")
                return True
            else:
                self._update_progress("init", "warning", "LLM not configured (no API key)", 0.7)
                logger.warning("LLM service created but not configured (missing API key)")
                return False

        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self._llm_initialized = True
            self._llm_service = None
            self._update_progress("init", "warning", f"LLM unavailable: {e}", 0.7)
            return False

    def _update_progress(
        self,
        stage: str,
        status: str,
        message: str,
        progress: float
    ):
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, status, message, progress)
        logger.info(f"[{stage}] {status}: {message} ({progress:.0%})")

    def _initialize_rag(self) -> bool:
        """
        Initialize RAG infrastructure.

        Returns:
            True if RAG initialized successfully, False otherwise
        """
        if self._rag_initialized:
            return self._retriever is not None

        self._update_progress("init", "running", "Initializing RAG infrastructure...", 0.0)

        try:
            # Get embedding provider
            embeddings = get_embedding_provider(
                provider_type=self.settings.embedding_provider,
                api_key=self.settings.voyage_api_key,
                huggingface_token=self.settings.huggingface_token,
                allow_mock_fallback=True
            )

            # Log which provider is being used
            logger.info(f"Using embedding provider: {embeddings.model_name}")

            # Initialize vector store
            persist_dir = str(self.settings.chroma_persist_dir)
            self._vector_store = VectorStore(
                embedding_provider=embeddings,
                persist_dir=persist_dir,
                collection_name=self.settings.chroma_collection_name
            )

            # Check if specs are already indexed
            if self._vector_store.count() == 0:
                self._update_progress("init", "running", "Indexing specifications...", 0.3)
                indexer = DocumentIndexer(self._vector_store)

                # Try to index from file first, fall back to hardcoded
                spec_path = self.settings.get_spec_path()
                if spec_path.exists():
                    logger.info(f"Indexing spec from file: {spec_path}")
                    indexer.index_dm_spec(str(spec_path), clear_existing=True)
                else:
                    logger.info("Spec file not found, using hardcoded rules")
                    indexer.index_dm_spec_hardcoded(clear_existing=True)

            # Create retriever
            self._retriever = SpecificationRetriever(self._vector_store)
            self._rag_initialized = True

            self._update_progress("init", "success", "RAG initialized", 0.5)
            return True

        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            self._rag_initialized = True  # Mark as attempted
            self._retriever = None
            self._update_progress("init", "warning", f"RAG unavailable: {e}", 0.5)
            return False

    def run(
        self,
        input_file: Optional[str] = None,
        input_df: Optional[pd.DataFrame] = None,
        trial_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_qc: bool = False
    ) -> PipelineResult:
        """
        Run the complete harmonization pipeline.

        Args:
            input_file: Path to input CSV/Excel file
            input_df: Or provide DataFrame directly
            trial_id: Trial identifier (extracted from filename if not provided)
            output_dir: Output directory (uses default if not provided)
            skip_qc: Skip QC stage if True

        Returns:
            PipelineResult with harmonized data and reports
        """
        start_time = time.time()
        result = PipelineResult(success=False)

        try:
            # Validate inputs
            if input_file is None and input_df is None:
                raise ValueError("Either input_file or input_df must be provided")

            # Initialize RAG if enabled
            if self.use_rag:
                self._initialize_rag()

            # Initialize LLM if enabled
            if self.use_llm:
                self._initialize_llm()

            # Setup output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = self.settings.output_dir
            output_path.mkdir(parents=True, exist_ok=True)

            # Create pipeline context
            context = PipelineContext()
            context.set("output_dir", str(output_path))

            # Stage 1: Ingest
            self._update_progress("ingest", "running", "Loading data...", 0.1)
            ingest_result = self._run_ingest(context, input_file, input_df, trial_id)
            result.stage_results["ingest"] = ingest_result

            if not ingest_result.success:
                result.errors.append(f"Ingest failed: {ingest_result.error}")
                return result

            # Stage 2: Map
            self._update_progress("map", "running", "Mapping columns...", 0.3)
            map_result = self._run_map(context)
            result.stage_results["map"] = map_result

            if not map_result.success:
                result.errors.append(f"Map failed: {map_result.error}")
                return result

            result.mapping_log = context.get("mapping_log")

            # Stage 3: Harmonize
            self._update_progress("harmonize", "running", "Harmonizing values...", 0.5)
            harmonize_result = self._run_harmonize(context)
            result.stage_results["harmonize"] = harmonize_result

            if not harmonize_result.success:
                result.errors.append(f"Harmonize failed: {harmonize_result.error}")
                return result

            result.lineage = context.get("lineage", [])

            # Stage 4: QC (optional)
            if not skip_qc:
                self._update_progress("qc", "running", "Running quality checks...", 0.7)
                qc_result = self._run_qc(context)
                result.stage_results["qc"] = qc_result

                if not qc_result.success:
                    # QC failures are warnings, not fatal errors
                    result.warnings.append(f"QC issues found: {qc_result.error}")

                result.qc_report = context.get("qc_report")

            # Stage 5: Review (optional, requires LLM)
            if self.enable_review and self.use_llm and self._llm_service and self._llm_service.is_configured:
                self._update_progress("review", "running", "Running LLM review...", 0.8)
                review_result = self._run_review(context, result.qc_report)
                result.stage_results["review"] = review_result

                if review_result.success:
                    result.review_result = review_result.data
                    if review_result.data and review_result.data.get("approval") == "rejected":
                        result.warnings.append(f"LLM review flagged issues: {review_result.data.get('reason', 'Unknown')}")
                else:
                    result.warnings.append(f"LLM review failed: {review_result.error}")

            # Gather results
            self._update_progress("finalize", "running", "Finalizing results...", 0.9)

            result.harmonized_data = context.get("df")
            result.success = True
            result.metadata = {
                "trial_id": context.get("trial_id"),
                "input_file": input_file,
                "output_dir": str(output_path),
                "rag_enabled": self._retriever is not None,
                "llm_enabled": self._llm_service is not None and self._llm_service.is_configured,
                "review_enabled": self.enable_review,
                "embedding_provider": self._vector_store.embedding_provider.model_name if self._vector_store else "none",
                "llm_model": self._llm_service.model if self._llm_service else "none",
                "llm_tokens_used": self._llm_service.total_tokens_used if self._llm_service else 0,
                "timestamp": datetime.now().isoformat(),
                "rows_processed": len(result.harmonized_data) if result.harmonized_data is not None else 0,
            }

            # Save outputs
            self._save_outputs(result, output_path, context.get("trial_id", "output"))

            self._update_progress("complete", "success", "Pipeline complete", 1.0)

        except Exception as e:
            logger.exception("Pipeline failed with unexpected error")
            result.errors.append(f"Unexpected error: {str(e)}")
            self._update_progress("error", "failed", str(e), 0.0)

        finally:
            result.execution_time_ms = int((time.time() - start_time) * 1000)

        return result

    def _run_ingest(
        self,
        context: PipelineContext,
        input_file: Optional[str],
        input_df: Optional[pd.DataFrame],
        trial_id: Optional[str]
    ) -> AgentResult:
        """Run the ingest stage."""
        if input_df is not None:
            # Direct DataFrame input
            context.set("df", input_df)
            context.set("trial_id", trial_id or "UNKNOWN")
            context.set("ingest_metadata", {
                "source": "direct_dataframe",
                "rows": len(input_df),
                "columns": list(input_df.columns)
            })
            return AgentResult(success=True, data={"rows": len(input_df)})

        # File input - use IngestAgent
        context.set("input_file", input_file)
        if trial_id:
            context.set("trial_id", trial_id)

        agent = IngestAgent()
        return agent.run(context)

    def _run_map(self, context: PipelineContext) -> AgentResult:
        """Run the mapping stage."""
        agent = MapAgent(retriever=self._retriever)
        return agent.run(context)

    def _run_harmonize(self, context: PipelineContext) -> AgentResult:
        """Run the harmonization stage with optional LLM fallback."""
        agent = HarmonizeAgent(
            retriever=self._retriever,
            llm_service=self._llm_service,
            use_llm_fallback=self.use_llm and self._llm_service is not None
        )
        return agent.run(context)

    def _run_qc(self, context: PipelineContext) -> AgentResult:
        """Run the QC stage."""
        agent = QCAgent(retriever=self._retriever)
        return agent.run(context)

    def _run_review(
        self,
        context: PipelineContext,
        qc_report: Optional[pd.DataFrame] = None
    ) -> AgentResult:
        """Run the LLM review stage."""
        # Pass QC issues to the review agent
        if qc_report is not None and len(qc_report) > 0:
            context.set("qc_issues", qc_report.to_dict(orient="records"))

        agent = ReviewAgent(
            llm_service=self._llm_service,
            retriever=self._retriever,
            sample_size=10,
            auto_approve_threshold="acceptable"
        )
        return agent.run(context)

    def _save_outputs(
        self,
        result: PipelineResult,
        output_dir: Path,
        trial_id: str
    ):
        """Save pipeline outputs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{trial_id}_{timestamp}"

        try:
            # Save harmonized data
            if result.harmonized_data is not None:
                output_file = output_dir / f"{prefix}_harmonized.csv"
                result.harmonized_data.to_csv(output_file, index=False)
                logger.info(f"Saved harmonized data: {output_file}")
                result.metadata["harmonized_file"] = str(output_file)

            # Save QC report
            if result.qc_report is not None and len(result.qc_report) > 0:
                qc_file = output_dir / f"{prefix}_qc_report.csv"
                result.qc_report.to_csv(qc_file, index=False)
                logger.info(f"Saved QC report: {qc_file}")
                result.metadata["qc_file"] = str(qc_file)

            # Save mapping log
            if result.mapping_log:
                import json
                mapping_file = output_dir / f"{prefix}_mapping_log.json"
                with open(mapping_file, 'w') as f:
                    json.dump(result.mapping_log, f, indent=2, default=str)
                logger.info(f"Saved mapping log: {mapping_file}")
                result.metadata["mapping_file"] = str(mapping_file)

            # Save lineage
            if result.lineage:
                import json
                lineage_file = output_dir / f"{prefix}_lineage.json"
                with open(lineage_file, 'w') as f:
                    json.dump(result.lineage, f, indent=2, default=str)
                logger.info(f"Saved lineage: {lineage_file}")
                result.metadata["lineage_file"] = str(lineage_file)

            # Save review result
            if result.review_result:
                import json
                review_file = output_dir / f"{prefix}_review.json"
                with open(review_file, 'w') as f:
                    json.dump(result.review_result, f, indent=2, default=str)
                logger.info(f"Saved review result: {review_file}")
                result.metadata["review_file"] = str(review_file)

        except Exception as e:
            logger.error(f"Error saving outputs: {e}")
            result.warnings.append(f"Failed to save some outputs: {e}")

    def get_retriever(self) -> Optional[SpecificationRetriever]:
        """Get the retriever instance (for direct queries)."""
        if not self._rag_initialized:
            self._initialize_rag()
        return self._retriever

    def reindex_specifications(self, spec_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Reindex specifications (useful after spec updates).

        Args:
            spec_path: Optional path to spec file (uses default if not provided)

        Returns:
            Indexing statistics
        """
        if not self._rag_initialized:
            self._initialize_rag()

        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")

        indexer = DocumentIndexer(self._vector_store)

        if spec_path:
            path = Path(spec_path)
        else:
            path = self.settings.get_spec_path()

        if path.exists():
            return indexer.index_dm_spec(str(path), clear_existing=True)
        else:
            return indexer.index_dm_spec_hardcoded(clear_existing=True)


def create_orchestrator(
    use_rag: bool = True,
    use_llm: bool = True,
    enable_review: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    **settings_overrides
) -> PipelineOrchestrator:
    """
    Factory function to create a configured orchestrator.

    Args:
        use_rag: Whether to use RAG for spec retrieval
        use_llm: Whether to use LLM for value resolution and review
        enable_review: Whether to run LLM review stage
        progress_callback: Optional progress callback
        **settings_overrides: Override any settings

    Returns:
        Configured PipelineOrchestrator
    """
    settings = get_settings(**settings_overrides)
    return PipelineOrchestrator(
        settings=settings,
        progress_callback=progress_callback,
        use_rag=use_rag,
        use_llm=use_llm,
        enable_review=enable_review
    )
