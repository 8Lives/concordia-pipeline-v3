"""
Review Agent - LLM-Powered Output Validation

Uses Claude to review harmonized data for quality and compliance.
Provides intelligent assessment beyond rule-based QC checks.

Key Features:
- Holistic data quality assessment
- Pattern recognition for subtle issues
- Contextual recommendations
- Approval/rejection decisions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .base import AgentBase, AgentConfig, AgentResult, PipelineContext, ProgressCallback

logger = logging.getLogger(__name__)


class ReviewAgent(AgentBase):
    """
    LLM-powered review agent for harmonized data validation.

    Uses Claude to:
    1. Review data quality holistically
    2. Identify subtle issues missed by rule-based QC
    3. Provide contextual recommendations
    4. Make approval/rejection decisions
    """

    def __init__(
        self,
        llm_service=None,
        retriever=None,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
        sample_size: int = 10,
        auto_approve_threshold: str = "good"
    ):
        """
        Initialize the Review Agent.

        Args:
            llm_service: LLM service for review
            retriever: Specification retriever for context
            config: Agent configuration
            progress_callback: Progress callback
            sample_size: Number of rows to sample for review
            auto_approve_threshold: Quality level for auto-approval
                ("good", "acceptable", or None for manual only)
        """
        super().__init__(
            name="review",
            config=config or AgentConfig(timeout_seconds=60, required=False),
            progress_callback=progress_callback
        )
        self.llm_service = llm_service
        self.retriever = retriever
        self.sample_size = sample_size
        self.auto_approve_threshold = auto_approve_threshold

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        # Check for harmonized_df first, fall back to df
        if context.get("harmonized_df") is None and context.get("df") is None:
            return "No DataFrame found in context (harmonized_df or df)"
        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute the review process."""
        try:
            # Use harmonized_df (from harmonize agent) if available, otherwise fall back to df
            # Use explicit None check to avoid DataFrame truth value ambiguity
            harmonized_df = context.get("harmonized_df")
            df = harmonized_df if harmonized_df is not None else context.get("df")
            qc_report = context.get("qc_report")
            mapping_log = context.get("mapping_log", [])
            lineage_log = context.get("harmonize_lineage_log", [])

            self._update_status(self._status, "Starting review...", 0.1)

            # Check if LLM is available
            if not self.llm_service or not self.llm_service.is_configured:
                logger.info("LLM not configured, performing basic review")
                return self._basic_review(context, df, qc_report)

            # Prepare data for LLM review
            self._update_status(self._status, "Preparing review data...", 0.2)
            data_sample = self._prepare_sample(df)
            column_stats = self._compute_column_stats(df)
            qc_issues = self._format_qc_issues(qc_report)
            spec_summary = self._get_spec_summary()
            variable_rules = self._get_variable_rules()

            # Call LLM for review
            self._update_status(self._status, "LLM reviewing data...", 0.4)
            response = self.llm_service.review_harmonized_data(
                data_sample=data_sample,
                variable_rules=variable_rules,
                qc_issues=qc_issues,
                column_stats=column_stats
            )

            if not response.success:
                logger.warning(f"LLM review failed: {response.error}")
                return self._basic_review(context, df, qc_report)

            # Process review results
            self._update_status(self._status, "Processing review results...", 0.8)
            review_result = response.parsed_data or {}

            # Extract stoplight (new v3 format)
            stoplight = review_result.get("stoplight") or review_result.get("approval", "UNKNOWN")
            stoplight = stoplight.upper() if stoplight else "UNKNOWN"

            # Determine approval status based on stoplight
            overall_quality = review_result.get("overall_quality", "unknown")

            # Auto-approval logic based on stoplight
            approved = stoplight == "GREEN"

            # Store review results with stoplight fields
            review_data = {
                "stoplight": stoplight,
                "approval": stoplight,  # Alias for compatibility
                "overall_quality": overall_quality,
                "approved": approved,
                "core_variables_present": review_result.get("core_variables_present", []),
                "core_variables_missing": review_result.get("core_variables_missing", []),
                "core_variables_count": review_result.get("core_variables_count", 0),
                "formatting_issues": review_result.get("formatting_issues", []),
                "critical_issues": review_result.get("critical_issues", []),
                "recommendations": review_result.get("recommendations", []),
                "reason": review_result.get("reason", "Review completed"),
                "llm_tokens_used": response.usage,
                "review_type": "llm"
            }

            context.set("review_result", review_data)
            context.set("review_metadata", {
                "sample_size": len(data_sample),
                "llm_model": response.model,
                "auto_approved": approved
            })

            self._update_status(self._status, "Review complete", 1.0)

            return AgentResult(
                success=True,
                data=review_data,
                metadata=context.get("review_metadata")
            )

        except Exception as e:
            logger.exception("Review agent failed")
            return AgentResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

    def _basic_review(
        self,
        context: PipelineContext,
        df: pd.DataFrame,
        qc_report: Optional[pd.DataFrame]
    ) -> AgentResult:
        """Perform basic review without LLM using STOPLIGHT rules."""
        # Check for core variables
        core_vars = ["SEX", "RACE", "ETHNIC", "COUNTRY"]
        age_vars = ["AGE", "AGEGP"]

        # Check which core variables are present and populated
        present = []
        missing = []

        for var in core_vars:
            if var in df.columns and df[var].notna().any():
                present.append(var)
            else:
                missing.append(var)

        # Check AGE/AGEGP (at least one needed)
        age_present = False
        for age_var in age_vars:
            if age_var in df.columns and df[age_var].notna().any():
                present.append(age_var)
                age_present = True
                break

        if not age_present:
            missing.append("AGE/AGEGP")

        # Determine stoplight based on rules
        core_count = len(present)
        missing_count = 5 - core_count

        if missing_count == 0:
            stoplight = "GREEN"
            overall_quality = "good"
        elif missing_count <= 2:
            stoplight = "YELLOW"
            overall_quality = "acceptable"
        else:
            stoplight = "RED"
            overall_quality = "poor"

        approved = stoplight == "GREEN"

        review_data = {
            "stoplight": stoplight,
            "approval": stoplight,
            "overall_quality": overall_quality,
            "approved": approved,
            "core_variables_present": present,
            "core_variables_missing": missing,
            "core_variables_count": core_count,
            "formatting_issues": [],
            "critical_issues": [],
            "recommendations": [f"Add missing core variables: {', '.join(missing)}"] if missing else [],
            "reason": f"Basic review: {core_count}/5 core variables present",
            "review_type": "basic"
        }

        context.set("review_result", review_data)

        return AgentResult(
            success=True,
            data=review_data,
            metadata={"review_type": "basic", "llm_available": False}
        )

    def _prepare_sample(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare a representative sample of the data."""
        if len(df) <= self.sample_size:
            sample = df
        else:
            # Stratified sample: first few, middle, last few
            n = self.sample_size
            indices = (
                list(range(min(n//3, len(df)))) +
                list(range(len(df)//2 - n//6, len(df)//2 + n//6)) +
                list(range(max(0, len(df) - n//3), len(df)))
            )
            indices = sorted(set(indices))[:n]
            sample = df.iloc[indices]

        return sample.to_dict(orient="records")

    def _compute_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute summary statistics for each column."""
        stats = {}
        for col in df.columns:
            col_stats = {
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(df[col].isna().sum() / len(df) * 100, 1) if len(df) > 0 else 0
            }

            # Add unique values for categorical columns
            if df[col].dtype == "object":
                unique_vals = df[col].dropna().unique()
                col_stats["unique_count"] = len(unique_vals)
                col_stats["sample_values"] = list(unique_vals[:5])
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_stats["min"] = float(df[col].min()) if df[col].notna().any() else None
                col_stats["max"] = float(df[col].max()) if df[col].notna().any() else None

            stats[col] = col_stats

        return stats

    def _format_qc_issues(self, qc_report: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Format QC issues for LLM review."""
        if qc_report is None or len(qc_report) == 0:
            return []

        return qc_report.head(20).to_dict(orient="records")

    def _get_spec_summary(self) -> str:
        """Get specification summary for context."""
        summary = """DM (Demographics) Domain Harmonization:
- Required: TRIAL, SUBJID, SEX, RACE
- Conditional: AGE (if available), AGEGP (if AGE not available)
- Optional: ETHNIC, COUNTRY, SITEID, STUDYID, USUBJID, ARM, ARMCD, BRTHDTC, RFSTDTC, RFENDTC
- DOMAIN: Always 'DM'
- Dates: ISO 8601 format (YYYY-MM-DD)
- Text: Mixed case where applicable"""

        return summary

    def _get_variable_rules(self) -> Dict[str, str]:
        """Get variable rules from retriever or fallback."""
        rules = {
            "TRIAL": "NCT format, extracted from filename",
            "SUBJID": "Unique subject identifier, trimmed",
            "SEX": "Male, Female, or Unknown",
            "RACE": "CDISC controlled terminology",
            "AGE": "Numeric, 0-120, derived from dates if missing",
            "ETHNIC": "Hispanic or Latino, Not Hispanic or Latino, Unknown",
            "COUNTRY": "Full country name",
            "DOMAIN": "Always 'DM'"
        }

        if self.retriever:
            try:
                for var in rules.keys():
                    spec = self.retriever.get_variable_rules(var)
                    if spec and spec.transformation:
                        rules[var] = spec.transformation
            except Exception as e:
                logger.warning(f"Could not retrieve rules from RAG: {e}")

        return rules
