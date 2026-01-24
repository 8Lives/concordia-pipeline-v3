"""
QC Agent - RAG-Enhanced Quality Control

Performs quality checks on harmonized data using RAG-retrieved specifications.
Each QC issue can be traced back to the specification rule that defines it.

Key Changes from v2:
- QC rules retrieved from RAG with spec citations
- Required variables list retrieved from RAG
- Coded variable identification from RAG
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .base import AgentBase, AgentConfig, AgentResult, PipelineContext, ProgressCallback
from utils.helpers import validate_nct_format, is_full_date

logger = logging.getLogger(__name__)


# Fallback QC issue definitions
FALLBACK_QC_ISSUES = {
    "TRIAL_MISSING_OR_INVALID": "TRIAL is missing or does not match NCT format",
    "DUPLICATE_SUBJECT": "Duplicate (TRIAL, SUBJID) combination",
    "MISSING_REQUIRED_VALUE": "Required variable has missing value",
    "MISSING_AGE_AND_AGEGP": "Both AGE and AGEGP are missing",
    "CODED_VALUE_NO_DICTIONARY": "Coded value requires dictionary but none provided",
    "DATE_INVALID": "Date value is not parseable",
    "DATE_ORDER_INVALID": "RFENDTC is before RFSTDTC",
    "AGE_INCONSISTENT_WITH_DATES": "Derived AGE differs from provided AGE by >2 years",
    "SUBJID_MAPPING_SUSPECT": "SUBJID mapping may be incorrect",
    "COLUMN_MAPPING_HEURISTIC": "Column mapped using heuristic rather than name matching",
}

FALLBACK_REQUIRED_VARIABLES = ["TRIAL", "SUBJID", "SEX", "RACE"]
FALLBACK_CODED_VARIABLES = ["SEX", "RACE", "ETHNIC", "ARMCD"]
FALLBACK_DATE_VARIABLES = ["BRTHDTC", "RFSTDTC", "RFENDTC"]


class QCAgent(AgentBase):
    """
    Performs quality control checks on harmonized data.

    Uses RAG to retrieve:
    - QC rule definitions with spec citations
    - Required variable list
    - Coded variable identification
    - Validation thresholds
    """

    def __init__(
        self,
        retriever=None,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """
        Initialize the QC Agent.

        Args:
            retriever: SpecificationRetriever for RAG lookups
            config: Agent configuration
            progress_callback: Progress callback
        """
        super().__init__(
            name="qc",
            config=config or AgentConfig(timeout_seconds=120),
            progress_callback=progress_callback
        )
        self.retriever = retriever
        self._qc_rules_cache: Optional[List[Dict]] = None
        self._required_vars_cache: Optional[List[str]] = None

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        if context.get("harmonized_df") is None:
            return "No harmonized DataFrame found in context"
        return None

    def _get_required_variables(self) -> List[str]:
        """Get required variables from RAG or fallback."""
        if self._required_vars_cache is not None:
            return self._required_vars_cache

        required = []

        if self.retriever:
            try:
                required = self.retriever.get_required_variables()
                if required:
                    logger.debug(f"RAG: Required variables: {required}")
            except Exception as e:
                logger.warning(f"RAG lookup failed for required variables: {e}")

        if not required:
            required = FALLBACK_REQUIRED_VARIABLES

        self._required_vars_cache = required
        return required

    def _get_qc_rules(self) -> List[Dict[str, Any]]:
        """Get QC rules from RAG or fallback."""
        if self._qc_rules_cache is not None:
            return self._qc_rules_cache

        rules = []

        if self.retriever:
            try:
                rules = self.retriever.get_qc_rules()
                if rules:
                    logger.debug(f"RAG: Retrieved {len(rules)} QC rules")
            except Exception as e:
                logger.warning(f"RAG lookup failed for QC rules: {e}")

        if not rules:
            # Convert fallback to list format
            rules = [
                {"issue_type": k, "description": v, "spec_reference": "DM_Harmonization_Spec_v1.4"}
                for k, v in FALLBACK_QC_ISSUES.items()
            ]

        self._qc_rules_cache = rules
        return rules

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute quality control checks."""
        try:
            df = context.get("harmonized_df")
            mapping_log = context.get("mapping_log", [])
            lineage_log = context.get("harmonize_lineage_log", [])
            trial_id = context.get("trial_id")
            dictionary = context.get("dictionary", {})

            self._update_status(self._status, "Starting QC checks...", 0.1)

            issues = []

            # Check TRIAL validity
            self._update_status(self._status, "Checking TRIAL validity...", 0.2)
            issues.extend(self._check_trial_validity(df, trial_id))

            # Check uniqueness
            self._update_status(self._status, "Checking uniqueness...", 0.3)
            issues.extend(self._check_uniqueness(df))

            # Check required values
            self._update_status(self._status, "Checking required values...", 0.4)
            issues.extend(self._check_required_values(df))

            # Check age completeness
            self._update_status(self._status, "Checking age completeness...", 0.5)
            issues.extend(self._check_age_completeness(df))

            # Check coded values
            self._update_status(self._status, "Checking coded values...", 0.6)
            issues.extend(self._check_coded_values(df, dictionary))

            # Check date validity
            self._update_status(self._status, "Checking date validity...", 0.7)
            issues.extend(self._check_date_validity(df))

            # Check mapping quality
            self._update_status(self._status, "Checking mapping quality...", 0.8)
            issues.extend(self._check_mapping_quality(mapping_log))

            # Build QC report
            self._update_status(self._status, "Building QC report...", 0.9)
            qc_report = self._build_qc_report(issues, trial_id)

            # Build transformation summary
            transformation_summary = self._build_transformation_summary(
                mapping_log, lineage_log
            )

            # Store results
            context.set("qc_report", qc_report)
            context.set("transformation_summary", transformation_summary)
            context.set("qc_metadata", {
                "total_issues": len(issues),
                "issues_by_type": self._count_issues_by_type(issues),
                "rag_enabled": self.retriever is not None,
            })

            return AgentResult(
                success=True,
                data={
                    "qc_report": qc_report,
                    "transformation_summary": transformation_summary,
                },
                metadata=context.get("qc_metadata")
            )

        except Exception as e:
            logger.exception("QC agent failed")
            return AgentResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

    def _check_trial_validity(
        self,
        df: pd.DataFrame,
        trial_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check TRIAL matches NCT format."""
        issues = []

        if "TRIAL" not in df.columns:
            issues.append({
                "issue_type": "TRIAL_MISSING_OR_INVALID",
                "variable": "TRIAL",
                "n_rows_affected": len(df),
                "example_values": [],
                "notes": "TRIAL column not found",
                "spec_reference": self._get_spec_reference("TRIAL_MISSING_OR_INVALID"),
            })
            return issues

        # Check each unique value
        unique_trials = df["TRIAL"].dropna().unique()

        for trial in unique_trials:
            if not validate_nct_format(str(trial)):
                affected = df[df["TRIAL"] == trial]
                issues.append({
                    "issue_type": "TRIAL_MISSING_OR_INVALID",
                    "variable": "TRIAL",
                    "n_rows_affected": len(affected),
                    "example_values": [str(trial)],
                    "notes": f"Value '{trial}' does not match NCT format (^NCT\\d{{8}}$)",
                    "spec_reference": self._get_spec_reference("TRIAL_MISSING_OR_INVALID"),
                })

        # Check for missing TRIAL
        missing = df["TRIAL"].isna().sum()
        if missing > 0:
            issues.append({
                "issue_type": "TRIAL_MISSING_OR_INVALID",
                "variable": "TRIAL",
                "n_rows_affected": int(missing),
                "example_values": [],
                "notes": "TRIAL is blank/null",
                "spec_reference": self._get_spec_reference("TRIAL_MISSING_OR_INVALID"),
            })

        return issues

    def _check_uniqueness(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check (TRIAL, SUBJID) uniqueness."""
        issues = []

        if "TRIAL" not in df.columns or "SUBJID" not in df.columns:
            return issues

        # Find duplicates
        duplicates = df[df.duplicated(subset=["TRIAL", "SUBJID"], keep=False)]

        if len(duplicates) > 0:
            # Get example duplicate pairs
            example_pairs = duplicates.groupby(["TRIAL", "SUBJID"]).size().head(5)
            examples = [f"{t}-{s}" for t, s in example_pairs.index]

            issues.append({
                "issue_type": "DUPLICATE_SUBJECT",
                "variable": "TRIAL, SUBJID",
                "n_rows_affected": len(duplicates),
                "example_values": examples,
                "notes": f"{len(duplicates)} rows have duplicate (TRIAL, SUBJID) combinations",
                "spec_reference": self._get_spec_reference("DUPLICATE_SUBJECT"),
            })

            # Check if ALL rows are duplicates (suggests mapping error)
            if len(duplicates) == len(df):
                issues.append({
                    "issue_type": "SUBJID_MAPPING_SUSPECT",
                    "variable": "SUBJID",
                    "n_rows_affected": len(df),
                    "example_values": list(df["SUBJID"].head(5).astype(str)),
                    "notes": "100% of rows are duplicates - SUBJID mapping likely incorrect",
                    "spec_reference": self._get_spec_reference("SUBJID_MAPPING_SUSPECT"),
                })

        return issues

    def _check_required_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check required variables have values."""
        issues = []
        required = self._get_required_variables()

        for var in required:
            if var not in df.columns:
                issues.append({
                    "issue_type": "MISSING_REQUIRED_VALUE",
                    "variable": var,
                    "n_rows_affected": len(df),
                    "example_values": [],
                    "notes": f"Required variable {var} not in output",
                    "spec_reference": self._get_spec_reference("MISSING_REQUIRED_VALUE"),
                })
                continue

            missing = df[var].isna().sum()
            if missing > 0:
                # Get some example row indices
                missing_rows = df[df[var].isna()].index[:5].tolist()

                issues.append({
                    "issue_type": "MISSING_REQUIRED_VALUE",
                    "variable": var,
                    "n_rows_affected": int(missing),
                    "example_values": [f"row {r}" for r in missing_rows],
                    "notes": f"Required variable {var} has {missing} missing values",
                    "spec_reference": self._get_spec_reference("MISSING_REQUIRED_VALUE"),
                })

        return issues

    def _check_age_completeness(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check that AGE or AGEGP is present."""
        issues = []

        has_age = "AGE" in df.columns
        has_agegp = "AGEGP" in df.columns

        if not has_age and not has_agegp:
            issues.append({
                "issue_type": "MISSING_AGE_AND_AGEGP",
                "variable": "AGE, AGEGP",
                "n_rows_affected": len(df),
                "example_values": [],
                "notes": "Neither AGE nor AGEGP columns present",
                "spec_reference": self._get_spec_reference("MISSING_AGE_AND_AGEGP"),
            })
            return issues

        # Check rows where both are missing
        if has_age and has_agegp:
            both_missing = df[df["AGE"].isna() & df["AGEGP"].isna()]
        elif has_age:
            both_missing = df[df["AGE"].isna()]
        else:
            both_missing = df[df["AGEGP"].isna()]

        if len(both_missing) > 0:
            issues.append({
                "issue_type": "MISSING_AGE_AND_AGEGP",
                "variable": "AGE, AGEGP",
                "n_rows_affected": len(both_missing),
                "example_values": list(both_missing.index[:5]),
                "notes": f"{len(both_missing)} rows missing both AGE and AGEGP",
                "spec_reference": self._get_spec_reference("MISSING_AGE_AND_AGEGP"),
            })

        return issues

    def _check_coded_values(
        self,
        df: pd.DataFrame,
        dictionary: Dict
    ) -> List[Dict[str, Any]]:
        """Check if coded values need dictionary decoding."""
        issues = []
        coded_vars = FALLBACK_CODED_VARIABLES

        for var in coded_vars:
            if var not in df.columns:
                continue

            # Check if dictionary was available for this variable
            has_dict = var in dictionary

            # Look for values that appear to be codes (1-2 digit numbers)
            code_pattern = re.compile(r'^[0-9]{1,2}$')

            coded_values = df[var].dropna().apply(
                lambda x: bool(code_pattern.match(str(x).strip()))
            )
            n_coded = coded_values.sum()

            if n_coded > 0 and not has_dict:
                examples = df[var][coded_values].unique()[:5].tolist()
                issues.append({
                    "issue_type": "CODED_VALUE_NO_DICTIONARY",
                    "variable": var,
                    "n_rows_affected": int(n_coded),
                    "example_values": [str(e) for e in examples],
                    "notes": f"{n_coded} rows have coded values but no dictionary provided",
                    "spec_reference": self._get_spec_reference("CODED_VALUE_NO_DICTIONARY"),
                })

        return issues

    def _check_date_validity(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check date fields are valid and in correct order."""
        issues = []
        date_vars = FALLBACK_DATE_VARIABLES

        # Check each date field for validity
        for var in date_vars:
            if var not in df.columns:
                continue

            # Check for invalid dates
            def is_valid_date(x):
                if pd.isna(x):
                    return True
                val = str(x).strip()
                # Should be ISO format: YYYY, YYYY-MM, or YYYY-MM-DD
                if re.match(r'^\d{4}(-\d{2})?(-\d{2})?$', val):
                    return True
                # Or contain digits (might be convertible)
                return bool(re.search(r'\d', val))

            invalid = df[~df[var].apply(is_valid_date)]
            if len(invalid) > 0:
                examples = invalid[var].head(5).tolist()
                issues.append({
                    "issue_type": "DATE_INVALID",
                    "variable": var,
                    "n_rows_affected": len(invalid),
                    "example_values": [str(e) for e in examples],
                    "notes": f"{len(invalid)} rows have unparseable date values",
                    "spec_reference": self._get_spec_reference("DATE_INVALID"),
                })

        # Check date order: RFENDTC should be >= RFSTDTC
        if "RFSTDTC" in df.columns and "RFENDTC" in df.columns:
            def check_date_order(row):
                start = row.get("RFSTDTC")
                end = row.get("RFENDTC")

                if pd.isna(start) or pd.isna(end):
                    return True

                start_str = str(start).strip()
                end_str = str(end).strip()

                # Only compare if both are full dates
                if is_full_date(start_str) and is_full_date(end_str):
                    return end_str >= start_str

                return True

            order_issues = df[~df.apply(check_date_order, axis=1)]
            if len(order_issues) > 0:
                examples = []
                for _, row in order_issues.head(3).iterrows():
                    examples.append(f"start={row.get('RFSTDTC')}, end={row.get('RFENDTC')}")

                issues.append({
                    "issue_type": "DATE_ORDER_INVALID",
                    "variable": "RFSTDTC, RFENDTC",
                    "n_rows_affected": len(order_issues),
                    "example_values": examples,
                    "notes": "RFENDTC is before RFSTDTC",
                    "spec_reference": self._get_spec_reference("DATE_ORDER_INVALID"),
                })

        return issues

    def _check_mapping_quality(
        self,
        mapping_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for potential mapping issues."""
        issues = []

        for entry in mapping_log:
            var = entry.get("output_variable")
            details = entry.get("details", {})

            # Flag heuristic mappings
            if details.get("matched_via") == "heuristic":
                issues.append({
                    "issue_type": "COLUMN_MAPPING_HEURISTIC",
                    "variable": var,
                    "n_rows_affected": 0,
                    "example_values": [],
                    "notes": f"{var} was mapped using uniqueness heuristic - verify correct column",
                    "spec_reference": self._get_spec_reference("COLUMN_MAPPING_HEURISTIC"),
                })

            # Flag low uniqueness warnings
            if "warning" in details and "uniqueness" in details.get("warning", "").lower():
                issues.append({
                    "issue_type": "SUBJID_MAPPING_SUSPECT",
                    "variable": var,
                    "n_rows_affected": 0,
                    "example_values": [],
                    "notes": details.get("warning"),
                    "spec_reference": self._get_spec_reference("SUBJID_MAPPING_SUSPECT"),
                })

        return issues

    def _get_spec_reference(self, issue_type: str) -> Optional[str]:
        """Get specification reference for an issue type from RAG."""
        rules = self._get_qc_rules()

        for rule in rules:
            if rule.get("issue_type") == issue_type:
                return rule.get("spec_reference")

        return "DM_Harmonization_Spec_v1.4, Section 5"

    def _build_qc_report(
        self,
        issues: List[Dict[str, Any]],
        trial_id: Optional[str]
    ) -> pd.DataFrame:
        """Build QC report DataFrame."""
        if not issues:
            # Return empty report with correct schema
            return pd.DataFrame(columns=[
                "TRIAL", "issue_type", "variable", "n_rows_affected",
                "example_values", "notes", "spec_reference"
            ])

        report_data = []
        for issue in issues:
            report_data.append({
                "TRIAL": trial_id or "UNKNOWN",
                "issue_type": issue.get("issue_type"),
                "variable": issue.get("variable"),
                "n_rows_affected": issue.get("n_rows_affected", 0),
                "example_values": ", ".join(str(e) for e in issue.get("example_values", [])[:5]),
                "notes": issue.get("notes"),
                "spec_reference": issue.get("spec_reference"),
            })

        return pd.DataFrame(report_data)

    def _count_issues_by_type(
        self,
        issues: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count issues by type."""
        counts = {}
        for issue in issues:
            issue_type = issue.get("issue_type", "UNKNOWN")
            counts[issue_type] = counts.get(issue_type, 0) + 1
        return counts

    def _build_transformation_summary(
        self,
        mapping_log: List[Dict[str, Any]],
        lineage_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build transformation summary combining mapping and harmonization info."""
        summary = []

        # Create lookup for lineage entries
        lineage_by_var = {e.get("variable"): e for e in lineage_log}

        for mapping in mapping_log:
            var = mapping.get("output_variable")
            lineage = lineage_by_var.get(var, {})

            summary.append({
                "variable": var,
                "source_column": mapping.get("source_column"),
                "mapping_operation": mapping.get("operation"),
                "transformation": lineage.get("transformation", "None"),
                "rows_changed": lineage.get("rows_changed", 0),
                "percent_changed": round(lineage.get("percent_changed", 0), 2),
                "missing_count": mapping.get("null_count", 0),
                "spec_reference": lineage.get("spec_reference"),
            })

        return summary
