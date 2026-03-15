"""
Map Agent - RAG-Enhanced Column Mapping

Maps source columns to output schema using RAG-retrieved source priorities.
Replaces hardcoded SOURCE_PRIORITY lookups with dynamic retrieval.

Key Changes from v2:
- Source column synonyms retrieved from RAG instead of schema.py
- SUBJID exclusion patterns retrieved from RAG
- All mapping rules traceable to specification
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .base import AgentBase, AgentConfig, AgentResult, PipelineContext, ProgressCallback
from utils.helpers import (
    find_column_match,
    find_subject_id_heuristic,
    validate_subjid_mapping,
    normalize_whitespace,
    extract_trial_from_filename,
)

logger = logging.getLogger(__name__)


# Fallback source priorities if RAG is unavailable
# These match the DM_Harmonization_Spec_v1.4
FALLBACK_SOURCE_PRIORITY = {
    "TRIAL": ["TRIAL"],
    "SUBJID": ["SUBJID", "RSUBJID", "RUSUBJID", "SUBJ", "SUBJECT", "SUBJECTID",
               "PATID", "PATIENTID", "PTNO", "PTID", "PID", "PID_A"],
    "SEX": ["SEX", "SEXC", "GENDER", "SEXCD"],
    "RACE": ["RACE", "RACESC", "RACECD", "RACEN"],
    "AGE": ["AGE", "APTS", "AGEATSCREEN", "AGEATRAND"],
    "AGEU": ["AGEU", "AGEUNITS", "AGEUNIT"],
    "AGEGP": ["AGEGP", "AGEGRP", "AGE_GROUP", "AGE_BAND", "AGE_CAT", "AGEGR1"],
    "ETHNIC": ["ETHNIC", "ETHGRP", "ETHNICGRP"],
    "COUNTRY": ["COUNTRY", "CNTRY"],
    "SITEID": ["SITEID", "RSITEID", "SITE", "SITENO", "CENT", "CENTER"],
    "STUDYID": ["STUDYID", "STUDY", "PROTNO"],
    "USUBJID": ["USUBJID", "RUSUBJID"],
    "ARMCD": ["ARMCD", "TRTARMCD", "TRTCODE", "TRTN"],
    "ARM": ["ARM", "TRTARM", "TRTLONG", "ACTTRT", "TRT"],
    "BRTHDTC": ["BRTHDTC", "BIRTHDT", "DOB", "BRTHDT"],
    "RFSTDTC": ["RFSTDTC", "RFXSTDTC", "RANDDTC", "RANDDT"],
    "RFENDTC": ["RFENDTC", "RFXENDTC", "ENDDT"],
    "DOMAIN": [],
}

FALLBACK_SUBJID_EXCLUSIONS = [
    "FORMID", "FORM_ID", "VISITID", "VISIT_ID",
    "PROTID", "PROT_ID", "STUDYID", "STUDY_ID",
    "SITEID", "SITE_ID", "ARMID", "ARM_ID",
    "DOMAINID", "DOMAIN_ID", "TESTID", "TEST_ID",
    "SEQID", "SEQ_ID", "RECORDID", "RECORD_ID",
]


class MapAgent(AgentBase):
    """
    Maps source data columns to standardized output schema.

    Uses RAG to retrieve:
    - Source column synonyms for each output variable
    - Exclusion patterns for SUBJID matching
    - Fallback sources for derived variables
    """

    def __init__(
        self,
        retriever=None,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """
        Initialize the Map Agent.

        Args:
            retriever: SpecificationRetriever for RAG lookups (optional)
            config: Agent configuration
            progress_callback: Progress update callback
        """
        super().__init__(
            name="map",
            config=config or AgentConfig(timeout_seconds=60),
            progress_callback=progress_callback
        )
        self.retriever = retriever
        self._source_priority_cache: Dict[str, List[str]] = {}
        self._exclusion_patterns: Optional[List[str]] = None

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        if context.get("df") is None:
            return "No DataFrame found in context (df)"
        if not isinstance(context.get("df"), pd.DataFrame):
            return "Context 'df' is not a pandas DataFrame"
        return None

    def _get_source_priority(self, variable: str) -> List[str]:
        """
        Get source column candidates for a variable.

        First tries RAG retrieval, falls back to hardcoded list.
        Results are cached for performance.
        """
        if variable in self._source_priority_cache:
            return self._source_priority_cache[variable]

        sources = []

        # Try RAG retrieval
        if self.retriever:
            try:
                sources = self.retriever.get_source_columns(variable)
                if sources:
                    logger.debug(f"RAG: Source columns for {variable}: {sources}")
            except Exception as e:
                logger.warning(f"RAG lookup failed for {variable}: {e}")

        # Fallback to hardcoded
        if not sources:
            sources = FALLBACK_SOURCE_PRIORITY.get(variable, [])
            logger.debug(f"Fallback: Source columns for {variable}: {sources}")

        self._source_priority_cache[variable] = sources
        return sources

    def _get_subjid_exclusions(self) -> List[str]:
        """Get columns to exclude from SUBJID matching."""
        if self._exclusion_patterns is not None:
            return self._exclusion_patterns

        # Try RAG retrieval
        if self.retriever:
            try:
                # Search for exclusion patterns in specifications
                results = self.retriever.search(
                    "SUBJID column exclusion patterns columns to exclude from subject ID matching",
                    n_results=3
                )
                # For now, use fallback - could parse from results
            except Exception as e:
                logger.warning(f"RAG lookup for SUBJID exclusions failed: {e}")

        # Use fallback
        self._exclusion_patterns = FALLBACK_SUBJID_EXCLUSIONS
        return self._exclusion_patterns

    def _get_output_schema(self) -> List[str]:
        """Get the output schema from RAG or fallback."""
        if self.retriever:
            try:
                schema = self.retriever.get_output_schema()
                if schema:
                    return schema
            except Exception as e:
                logger.warning(f"RAG lookup for output schema failed: {e}")

        # Fallback
        return [
            "TRIAL", "SUBJID", "SEX", "RACE", "AGE", "AGEU", "AGEGP", "ETHNIC",
            "COUNTRY", "SITEID", "STUDYID", "USUBJID", "ARMCD", "ARM",
            "BRTHDTC", "RFSTDTC", "RFENDTC", "DOMAIN"
        ]

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute column mapping."""
        try:
            df = context.get("df")
            trial_id = context.get("trial_id")
            metadata = context.get("ingest_metadata", {})

            self._update_status(self._status, "Starting column mapping...", 0.1)

            # Get output schema
            output_schema = self._get_output_schema()
            logger.info(f"Output schema: {output_schema}")

            # Initialize mapping results
            mapping_log = []
            mapped_columns = {}

            # Map each variable
            total_vars = len(output_schema)
            for i, output_var in enumerate(output_schema):
                progress = 0.1 + (0.8 * (i / total_vars))
                self._update_status(self._status, f"Mapping {output_var}...", progress)

                result = self._map_variable(df, output_var, trial_id, metadata)
                mapping_log.append(result)

                if result.get("source_column"):
                    mapped_columns[output_var] = result["source_column"]

            # Build mapped DataFrame
            self._update_status(self._status, "Building output DataFrame...", 0.9)
            mapped_df = self._build_mapped_df(df, mapping_log, output_schema)

            # Store results in context
            context.set("df", mapped_df)
            context.set("mapping_log", mapping_log)
            context.set("map_metadata", {
                "variables_mapped": len([m for m in mapping_log if m.get("source_column")]),
                "variables_derived": len([m for m in mapping_log if m.get("operation") == "Derive"]),
                "variables_constant": len([m for m in mapping_log if m.get("operation") == "Constant"]),
                "variables_missing": len([m for m in mapping_log if m.get("operation") == "Missing"]),
                "rag_enabled": self.retriever is not None,
            })

            return AgentResult(
                success=True,
                data={"mapped_df": mapped_df, "mapping_log": mapping_log},
                metadata=context.get("map_metadata")
            )

        except Exception as e:
            logger.exception("Map agent failed")
            return AgentResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

    def _map_variable(
        self,
        df: pd.DataFrame,
        output_var: str,
        trial_id: Optional[str],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map a single output variable to a source column.

        Returns mapping log entry with:
        - output_variable: Target variable name
        - source_column: Matched source column (or None)
        - operation: Copy, Constant, Derive, Missing
        - details: Additional information
        - spec_reference: RAG citation if available
        """
        result = {
            "output_variable": output_var,
            "source_column": None,
            "operation": "Missing",
            "details": {},
            "non_null_count": 0,
            "null_count": 0,
            "spec_reference": None,
        }

        # Get source priority from RAG
        source_candidates = self._get_source_priority(output_var)

        # Special handling for specific variables
        if output_var == "TRIAL":
            return self._map_trial(df, trial_id, metadata, result)

        elif output_var == "DOMAIN":
            result["operation"] = "Constant"
            result["details"] = {"value": "DM"}
            return result

        elif output_var == "SUBJID":
            return self._map_subjid(df, source_candidates, result)

        elif output_var == "USUBJID":
            return self._map_usubjid(df, source_candidates, result)

        elif output_var == "AGE":
            return self._map_age(df, source_candidates, result)

        elif output_var == "AGEGP":
            return self._map_agegp(df, source_candidates, result)

        # Standard mapping: find matching column
        exclusions = self._get_subjid_exclusions() if output_var == "SUBJID" else []
        source_col = find_column_match(df, source_candidates, exclusions)

        if source_col:
            result["source_column"] = source_col
            result["operation"] = "Copy"
            result["non_null_count"] = int(df[source_col].notna().sum())
            result["null_count"] = int(df[source_col].isna().sum())
            result["details"] = {"matched_from": source_candidates}
        else:
            result["operation"] = "Missing"
            result["details"] = {"searched": source_candidates}

        return result

    def _map_trial(
        self,
        df: pd.DataFrame,
        trial_id: Optional[str],
        metadata: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map TRIAL variable - primarily from filename."""
        if trial_id:
            result["operation"] = "Constant"
            result["details"] = {"source": "filename", "value": trial_id}
        else:
            # Try to find in data
            for col in ["TRIAL", "STUDYID", "PROTNO"]:
                if col in df.columns:
                    result["source_column"] = col
                    result["operation"] = "Copy"
                    result["non_null_count"] = int(df[col].notna().sum())
                    result["null_count"] = int(df[col].isna().sum())
                    break
            else:
                result["operation"] = "Missing"
                result["details"] = {"warning": "No trial identifier found"}

        return result

    def _map_subjid(
        self,
        df: pd.DataFrame,
        source_candidates: List[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map SUBJID with exclusion patterns and heuristic fallback."""
        exclusions = self._get_subjid_exclusions()

        # Try name-based matching first
        source_col = find_column_match(df, source_candidates, exclusions)

        if source_col:
            # Validate the mapping
            validation = validate_subjid_mapping(df, source_col)
            result["source_column"] = source_col
            result["operation"] = "Copy"
            result["non_null_count"] = int(df[source_col].notna().sum())
            result["null_count"] = int(df[source_col].isna().sum())
            result["details"] = {
                "matched_from": source_candidates,
                "validation": validation
            }

            if validation.get("uniqueness_ratio", 1.0) < 0.8:
                result["details"]["warning"] = "Low uniqueness ratio - verify mapping"

        else:
            # Heuristic fallback: find column with highest uniqueness
            heuristic_col = find_subject_id_heuristic(df, exclusions)
            if heuristic_col:
                result["source_column"] = heuristic_col
                result["operation"] = "Copy"
                result["non_null_count"] = int(df[heuristic_col].notna().sum())
                result["null_count"] = int(df[heuristic_col].isna().sum())
                result["details"] = {
                    "matched_via": "heuristic",
                    "warning": "Mapped using uniqueness heuristic - verify correct column"
                }
            else:
                result["operation"] = "Missing"
                result["details"] = {"searched": source_candidates, "heuristic": "failed"}

        return result

    def _map_usubjid(
        self,
        df: pd.DataFrame,
        source_candidates: List[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map USUBJID - copy if exists, derive if not."""
        source_col = find_column_match(df, source_candidates)

        if source_col:
            result["source_column"] = source_col
            result["operation"] = "Copy"
            result["non_null_count"] = int(df[source_col].notna().sum())
            result["null_count"] = int(df[source_col].isna().sum())
        else:
            # Will be derived in harmonize stage
            result["operation"] = "Derive"
            result["details"] = {"derivation": "STUDYID||'-'||SUBJID or TRIAL||'-'||SUBJID"}

        return result

    def _map_age(
        self,
        df: pd.DataFrame,
        source_candidates: List[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map AGE - copy if exists, mark for derivation if derivable."""
        source_col = find_column_match(df, source_candidates)

        if source_col:
            result["source_column"] = source_col
            result["operation"] = "Copy"
            result["non_null_count"] = int(df[source_col].notna().sum())
            result["null_count"] = int(df[source_col].isna().sum())
        else:
            # Check if derivable from dates
            brthdtc_candidates = self._get_source_priority("BRTHDTC")
            rfstdtc_candidates = self._get_source_priority("RFSTDTC")

            has_birth = find_column_match(df, brthdtc_candidates) is not None
            has_ref = find_column_match(df, rfstdtc_candidates) is not None

            if has_birth and has_ref:
                result["operation"] = "Derive"
                result["details"] = {"derivation": "From BRTHDTC and RFSTDTC"}
            else:
                result["operation"] = "Missing"
                result["details"] = {"note": "Not derivable - missing date fields"}

        return result

    def _map_agegp(
        self,
        df: pd.DataFrame,
        source_candidates: List[str],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map AGEGP - only if AGE is not available."""
        # Check if AGE is available
        age_candidates = self._get_source_priority("AGE")
        has_age = find_column_match(df, age_candidates) is not None

        if has_age:
            # AGE available - AGEGP should be blank
            result["operation"] = "Blank"
            result["details"] = {"reason": "AGE is available"}
        else:
            # Try to find age group column
            source_col = find_column_match(df, source_candidates)
            if source_col:
                result["source_column"] = source_col
                result["operation"] = "Copy"
                result["non_null_count"] = int(df[source_col].notna().sum())
                result["null_count"] = int(df[source_col].isna().sum())
            else:
                result["operation"] = "Missing"
                result["details"] = {"searched": source_candidates}

        return result

    def _build_mapped_df(
        self,
        df: pd.DataFrame,
        mapping_log: List[Dict[str, Any]],
        output_schema: List[str]
    ) -> pd.DataFrame:
        """Build the mapped DataFrame from source data."""
        mapped_data = {}

        for entry in mapping_log:
            output_var = entry["output_variable"]
            source_col = entry.get("source_column")
            operation = entry.get("operation")

            if source_col and source_col in df.columns:
                # Copy and normalize whitespace
                mapped_data[output_var] = df[source_col].apply(
                    lambda x: normalize_whitespace(str(x)) if pd.notna(x) else None
                )
            elif operation == "Constant":
                value = entry.get("details", {}).get("value")
                # Explicit Series to avoid scalar-broadcast issues
                mapped_data[output_var] = pd.Series(
                    [value] * len(df), index=df.index
                )
            else:
                # Empty column - will be populated in harmonize stage if derivable
                mapped_data[output_var] = None

        mapped_df = pd.DataFrame(mapped_data)

        # Ensure column order matches schema
        for col in output_schema:
            if col not in mapped_df.columns:
                mapped_df[col] = None

        return mapped_df[output_schema]
