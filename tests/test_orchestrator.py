"""
Tests for the Pipeline Orchestrator.

Tests the end-to-end pipeline coordination including:
- RAG initialization
- Agent sequencing
- Error handling
- Output generation
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import PipelineOrchestrator, PipelineResult, create_orchestrator
from config.settings import Settings, reset_settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp(dir="/sessions/magical-eager-clarke/tmp")
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Create sample input data."""
    return pd.DataFrame({
        'USUBJID': ['STUDY-001-001', 'STUDY-001-002', 'STUDY-001-003'],
        'SUBJID': ['001', '002', '003'],
        'SEX': ['M', 'F', 'Male'],
        'AGE': [45, 32, 58],
        'RACE': ['White', 'BLACK', 'Asian'],
        'ETHNIC': ['Hispanic', 'Not Hispanic or Latino', 'Unknown'],
        'COUNTRY': ['USA', 'GBR', 'JPN'],
        'SITEID': ['SITE01', 'SITE02', 'SITE03'],
        'BRTHDTC': ['1979-05-15', '1992-08-22', '1966-03-10'],
        'RFSTDTC': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'RFENDTC': ['2024-06-15', '2024-06-16', '2024-06-17'],
        'ARM': ['Treatment A', 'Treatment B', 'Placebo'],
        'ARMCD': ['TRT-A', 'TRT-B', 'PBO'],
    })


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temp directories."""
    reset_settings()
    return Settings(
        embedding_provider="local",  # Will fall back to mock
        chroma_persist_dir=Path(temp_dir) / "chroma",
        output_dir=Path(temp_dir) / "output",
        knowledge_base_dir=Path(temp_dir) / "knowledge_base"
    )


class TestOrchestratorBasic:
    """Basic orchestrator tests."""

    def test_orchestrator_creation(self, test_settings):
        """Test orchestrator can be created."""
        orchestrator = PipelineOrchestrator(settings=test_settings)
        assert orchestrator is not None
        assert orchestrator.settings == test_settings

    def test_orchestrator_with_rag_disabled(self, test_settings, sample_data, temp_dir):
        """Test orchestrator runs with RAG disabled."""
        orchestrator = PipelineOrchestrator(
            settings=test_settings,
            use_rag=False
        )

        result = orchestrator.run(
            input_df=sample_data,
            trial_id="TEST-001",
            output_dir=temp_dir
        )

        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.harmonized_data is not None
        assert len(result.harmonized_data) == len(sample_data)

    def test_orchestrator_with_rag_enabled(self, test_settings, sample_data, temp_dir):
        """Test orchestrator runs with RAG enabled."""
        orchestrator = PipelineOrchestrator(
            settings=test_settings,
            use_rag=True
        )

        result = orchestrator.run(
            input_df=sample_data,
            trial_id="TEST-002",
            output_dir=temp_dir
        )

        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.harmonized_data is not None


class TestOrchestratorOutputs:
    """Test orchestrator output generation."""

    def test_harmonized_data_columns(self, test_settings, sample_data, temp_dir):
        """Test harmonized data has expected columns."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=True)
        result = orchestrator.run(input_df=sample_data, trial_id="TEST-003", output_dir=temp_dir)

        assert result.success
        df = result.harmonized_data

        # Check required columns exist
        expected_cols = ["TRIAL", "SUBJID", "SEX", "RACE", "DOMAIN"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_qc_report_generated(self, test_settings, sample_data, temp_dir):
        """Test QC report is generated."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=True)
        result = orchestrator.run(input_df=sample_data, trial_id="TEST-004", output_dir=temp_dir)

        assert result.success
        # QC report should be a DataFrame (may be empty)
        assert result.qc_report is not None or "qc" in result.stage_results

    def test_mapping_log_generated(self, test_settings, sample_data, temp_dir):
        """Test mapping log is generated."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=True)
        result = orchestrator.run(input_df=sample_data, trial_id="TEST-005", output_dir=temp_dir)

        assert result.success
        assert result.mapping_log is not None
        assert len(result.mapping_log) > 0

    def test_output_files_saved(self, test_settings, sample_data, temp_dir):
        """Test output files are saved to disk."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=True)
        result = orchestrator.run(input_df=sample_data, trial_id="TEST-006", output_dir=temp_dir)

        assert result.success

        # Check files were created
        output_path = Path(temp_dir)
        csv_files = list(output_path.glob("*_harmonized.csv"))
        assert len(csv_files) > 0, "No harmonized CSV file found"


class TestOrchestratorErrorHandling:
    """Test orchestrator error handling."""

    def test_missing_input_error(self, test_settings, temp_dir):
        """Test error when no input provided."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=False)

        result = orchestrator.run(output_dir=temp_dir)

        # Should fail with appropriate error
        assert not result.success
        assert len(result.errors) > 0
        assert any("input_file or input_df" in err for err in result.errors)

    def test_empty_dataframe(self, test_settings, temp_dir):
        """Test handling of empty DataFrame."""
        orchestrator = PipelineOrchestrator(settings=test_settings, use_rag=False)
        empty_df = pd.DataFrame()

        result = orchestrator.run(input_df=empty_df, trial_id="TEST-EMPTY", output_dir=temp_dir)

        # Should fail gracefully
        assert not result.success or len(result.warnings) > 0


class TestOrchestratorProgress:
    """Test orchestrator progress callbacks."""

    def test_progress_callback_called(self, test_settings, sample_data, temp_dir):
        """Test progress callback is invoked."""
        progress_calls = []

        def progress_callback(stage, status, message, progress):
            progress_calls.append({
                "stage": stage,
                "status": status,
                "message": message,
                "progress": progress
            })

        orchestrator = PipelineOrchestrator(
            settings=test_settings,
            progress_callback=progress_callback,
            use_rag=True
        )

        result = orchestrator.run(input_df=sample_data, trial_id="TEST-PROG", output_dir=temp_dir)

        assert result.success
        assert len(progress_calls) > 0

        # Check we got progress for main stages
        stages = [p["stage"] for p in progress_calls]
        assert "init" in stages or "ingest" in stages


class TestOrchestratorFactory:
    """Test the create_orchestrator factory function."""

    def test_create_orchestrator_default(self):
        """Test factory with default settings."""
        reset_settings()
        orchestrator = create_orchestrator()
        assert orchestrator is not None
        assert orchestrator.use_rag is True

    def test_create_orchestrator_no_rag(self):
        """Test factory with RAG disabled."""
        reset_settings()
        orchestrator = create_orchestrator(use_rag=False)
        assert orchestrator.use_rag is False

    def test_create_orchestrator_with_overrides(self, temp_dir):
        """Test factory with setting overrides."""
        reset_settings()
        orchestrator = create_orchestrator(
            use_rag=True,
            output_dir=temp_dir
        )
        assert str(orchestrator.settings.output_dir) == temp_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
