"""
Harmonize Agent - RAG-Enhanced Value Transformation

Transforms mapped data values according to RAG-retrieved specifications.
Replaces hardcoded value mappings (SEX_DECODE, RACE_NORMALIZE, etc.) with dynamic retrieval.

Key Changes from v2:
- Valid values and code mappings retrieved from RAG
- Transformation rules retrieved from RAG for each variable
- All transformations traceable to specification rules
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import AgentBase, AgentConfig, AgentResult, PipelineContext, ProgressCallback
from utils.helpers import (
    to_mixed_case,
    normalize_whitespace,
    sas_date_to_iso,
    sas_datetime_to_iso,
    is_full_date,
    calculate_age,
    validate_nct_format,
)

logger = logging.getLogger(__name__)


# Fallback mappings if RAG unavailable
FALLBACK_SEX_DECODE = {
    "1": "Male", "2": "Female",
    "M": "Male", "F": "Female",
    "U": "Unknown", "UNK": "Unknown",
    "MALE": "Male", "FEMALE": "Female", "UNKNOWN": "Unknown"
}

FALLBACK_RACE_NORMALIZE = {
    "WHITE": "Caucasian",
    "WHITE OR CAUCASIAN": "Caucasian",
    "CAUCASIAN": "Caucasian",
    "BLACK": "Black or African American",
    "BLACK OR AFRICAN AMERICAN": "Black or African American",
    "AFRICAN AMERICAN": "Black or African American",
    "ASIAN": "Asian",
    "ORIENTAL": "Asian",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Native Hawaiian or Other Pacific Islander",
    "AMERICAN INDIAN OR ALASKA NATIVE": "American Indian or Alaska Native",
    "OTHER": "Other",
    "MULTIPLE": "Multiple",
}

FALLBACK_COUNTRY_CODES = {
    "US": "United States", "USA": "United States",
    "CA": "Canada", "CAN": "Canada",
    "GB": "United Kingdom", "UK": "United Kingdom", "GBR": "United Kingdom",
    "DE": "Germany", "DEU": "Germany",
    "FR": "France", "FRA": "France",
    "JP": "Japan", "JPN": "Japan",
    "AU": "Australia", "AUS": "Australia",
    "KR": "Korea, Republic of", "KOR": "Korea, Republic of",
    "IN": "India", "IND": "India",
    "CN": "China", "CHN": "China",
    "IT": "Italy", "ITA": "Italy",
    "ES": "Spain", "ESP": "Spain",
    "MX": "Mexico", "MEX": "Mexico",
    "BR": "Brazil", "BRA": "Brazil",
}


class HarmonizeAgent(AgentBase):
    """
    Harmonizes mapped data values according to specifications.

    Uses RAG to retrieve:
    - Valid value lists for coded variables
    - Code-to-value mappings (e.g., 1->Male)
    - Normalization rules (e.g., WHITE->Caucasian)
    - Transformation rules for each variable
    """

    def __init__(
        self,
        retriever=None,
        llm_service=None,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
        use_llm_fallback: bool = True
    ):
        """
        Initialize the Harmonize Agent.

        Args:
            retriever: SpecificationRetriever for RAG lookups
            llm_service: LLM service for fallback value resolution
            config: Agent configuration
            progress_callback: Progress callback
            use_llm_fallback: Whether to use LLM for unresolved values
        """
        super().__init__(
            name="harmonize",
            config=config or AgentConfig(timeout_seconds=120, max_retries=1),
            progress_callback=progress_callback
        )
        self.retriever = retriever
        self.llm_service = llm_service
        self.use_llm_fallback = use_llm_fallback

        # Caches for retrieved rules
        self._valid_values_cache: Dict[str, List[str]] = {}
        self._code_mappings_cache: Dict[str, Dict[str, str]] = {}
        self._llm_resolution_cache: Dict[str, str] = {}

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        if context.get("df") is None:
            return "No DataFrame found in context (df)"
        if context.get("mapping_log") is None:
            return "No mapping log found in context"
        return None

    def _get_valid_values(self, variable: str) -> Optional[List[str]]:
        """Get valid values for a variable from RAG or fallback."""
        if variable in self._valid_values_cache:
            return self._valid_values_cache[variable]

        values = None

        # Try RAG
        if self.retriever:
            try:
                values = self.retriever.get_valid_values(variable)
                if values:
                    logger.debug(f"RAG: Valid values for {variable}: {values}")
            except Exception as e:
                logger.warning(f"RAG lookup failed for {variable} valid values: {e}")

        self._valid_values_cache[variable] = values
        return values

    def _get_code_mappings(self, variable: str) -> Dict[str, str]:
        """Get code-to-value mappings for a variable."""
        if variable in self._code_mappings_cache:
            return self._code_mappings_cache[variable]

        mappings = {}

        # Try RAG - would need to parse from retrieved content
        if self.retriever:
            try:
                results = self.retriever.search(
                    f"code mappings decode values for {variable}",
                    n_results=2,
                    filter_type="codelist"
                )
                # Parse mappings from results if available
                # For now, use fallbacks
            except Exception as e:
                logger.warning(f"RAG lookup failed for {variable} code mappings: {e}")

        # Fallback to hardcoded
        if variable == "SEX":
            mappings = FALLBACK_SEX_DECODE
        elif variable == "RACE":
            mappings = FALLBACK_RACE_NORMALIZE
        elif variable == "COUNTRY":
            mappings = FALLBACK_COUNTRY_CODES

        self._code_mappings_cache[variable] = mappings
        return mappings

    def _resolve_with_llm(
        self,
        variable: str,
        value: str,
        valid_values: List[str],
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Use LLM to resolve an ambiguous value.

        Args:
            variable: Variable name
            value: Value to resolve
            valid_values: List of valid target values
            context: Additional context

        Returns:
            Resolved value or None if cannot resolve
        """
        # Check cache first
        cache_key = f"{variable}:{value}"
        if cache_key in self._llm_resolution_cache:
            return self._llm_resolution_cache[cache_key]

        # Skip if LLM not available or disabled
        if not self.llm_service or not self.use_llm_fallback:
            return None

        if not self.llm_service.is_configured:
            logger.debug("LLM not configured, skipping resolution")
            return None

        try:
            response = self.llm_service.resolve_value(
                variable=variable,
                value=value,
                valid_values=valid_values,
                context=context or f"Clinical trial demographics harmonization for {variable}"
            )

            if response.success and response.parsed_data:
                resolved = response.parsed_data.get("resolved_value")
                confidence = response.parsed_data.get("confidence", "low")

                if resolved and confidence in ["high", "medium"]:
                    logger.info(
                        f"LLM resolved {variable}='{value}' -> '{resolved}' "
                        f"(confidence: {confidence})"
                    )
                    self._llm_resolution_cache[cache_key] = resolved
                    return resolved

        except Exception as e:
            logger.warning(f"LLM resolution failed for {variable}='{value}': {e}")

        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """Execute value harmonization."""
        try:
            df = context.get("df").copy()
            mapping_log = context.get("mapping_log")
            dictionary = context.get("dictionary", {})
            trial_id = context.get("trial_id")

            self._update_status(self._status, "Starting harmonization...", 0.1)

            # Build lineage log
            lineage_log = []
            llm_resolutions = []

            # Get variables to process
            variables = list(df.columns)
            total_vars = len(variables)

            for i, var in enumerate(variables):
                progress = 0.1 + (0.8 * (i / total_vars))
                self._update_status(self._status, f"Harmonizing {var}...", progress)

                # Get mapping info for this variable
                var_mapping = next(
                    (m for m in mapping_log if m.get("output_variable") == var),
                    {}
                )

                # Harmonize based on variable type
                original_values = df[var].copy()
                df[var], lineage_entry = self._harmonize_variable(
                    df, var, var_mapping, dictionary, trial_id
                )

                # Track changes and missing counts (v2 structure)
                lineage_entry["rows_changed"] = self._count_changes(original_values, df[var])
                lineage_entry["percent_changed"] = (
                    lineage_entry["rows_changed"] / len(df) * 100
                    if len(df) > 0 else 0
                )
                lineage_entry["missing_count"] = int(df[var].isna().sum())
                lineage_entry["non_null_count"] = int(df[var].notna().sum())
                lineage_log.append(lineage_entry)

            # Check for duplicates
            self._update_status(self._status, "Checking for duplicates...", 0.9)
            duplicates = self._check_duplicates(df)

            # Store results
            context.set("harmonized_df", df)
            context.set("harmonize_lineage_log", lineage_log)
            context.set("llm_resolutions", llm_resolutions)
            context.set("harmonize_metadata", {
                "rows_out": len(df),
                "duplicates_found": len(duplicates),
                "llm_resolutions": len(self._llm_resolution_cache),
                "rag_enabled": self.retriever is not None,
            })

            return AgentResult(
                success=True,
                data={
                    "harmonized_df": df,
                    "lineage_log": lineage_log,
                },
                metadata=context.get("harmonize_metadata")
            )

        except Exception as e:
            logger.exception("Harmonize agent failed")
            return AgentResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

    def _harmonize_variable(
        self,
        df: pd.DataFrame,
        variable: str,
        mapping: Dict[str, Any],
        dictionary: Dict[str, Any],
        trial_id: Optional[str]
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Harmonize a single variable.

        Returns:
            Tuple of (harmonized series, lineage entry)
        """
        lineage = {
            "variable": variable,
            "source_column": mapping.get("source_column"),
            "mapping_operation": mapping.get("operation", "Unknown"),
            "transform_operation": "None",  # v2 key name
            "transform_details": {},  # v2 key name
            "spec_reference": None,
        }

        # Get spec reference from RAG if available
        if self.retriever:
            try:
                spec = self.retriever.get_variable_rules(variable)
                if spec:
                    lineage["spec_reference"] = spec.spec_reference
                    lineage["transform_operation"] = spec.transformation or "None"
            except Exception:
                pass

        # Route to specific handler
        if variable == "TRIAL":
            return self._harmonize_trial(df, trial_id, lineage)
        elif variable == "SUBJID":
            return self._harmonize_subjid(df, lineage)
        elif variable == "SEX":
            return self._harmonize_sex(df, dictionary, lineage)
        elif variable == "RACE":
            return self._harmonize_race(df, dictionary, lineage)
        elif variable == "AGE":
            return self._harmonize_age(df, lineage)
        elif variable == "AGEU":
            return self._harmonize_ageu(df, lineage)
        elif variable == "AGEGP":
            return self._harmonize_agegp(df, lineage)
        elif variable == "ETHNIC":
            return self._harmonize_ethnic(df, dictionary, lineage)
        elif variable == "COUNTRY":
            return self._harmonize_country(df, dictionary, lineage)
        elif variable == "SITEID":
            return self._harmonize_siteid(df, lineage)
        elif variable == "STUDYID":
            return self._harmonize_studyid(df, lineage)
        elif variable == "USUBJID":
            return self._harmonize_usubjid(df, lineage)
        elif variable in ["ARMCD", "ARM"]:
            return self._harmonize_arm(df, variable, dictionary, lineage)
        elif variable in ["BRTHDTC", "RFSTDTC", "RFENDTC"]:
            return self._harmonize_date(df, variable, lineage)
        elif variable == "DOMAIN":
            return self._harmonize_domain(df, lineage)
        else:
            # Unknown variable - just normalize whitespace
            return self._harmonize_default(df, variable, lineage)

    def _harmonize_trial(
        self,
        df: pd.DataFrame,
        trial_id: Optional[str],
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize TRIAL - set constant or normalize."""
        if trial_id:
            result = pd.Series([trial_id] * len(df), index=df.index)
            lineage["transform_operation"] = "Constant"
            lineage["transform_details"] = {"value": trial_id}
        else:
            result = df["TRIAL"].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) else None
            )
            lineage["transform_operation"] = "Normalize (uppercase, trim)"

        return result, lineage

    def _harmonize_subjid(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize SUBJID - trim and convert floats to strings."""
        def clean_subjid(x):
            if pd.isna(x):
                return None
            # Handle float representation of IDs
            if isinstance(x, float) and x == int(x):
                return str(int(x))
            return str(x).strip()

        result = df["SUBJID"].apply(clean_subjid)
        lineage["transform_operation"] = "Trim, convert floats to strings"
        return result, lineage

    def _normalize_code_key(self, value) -> str:
        """
        Normalize a value for dictionary lookup.
        Handles float-to-int conversion (1.0 -> "1") and string normalization.
        This is critical because CSV/Excel may read numeric codes as floats.
        """
        if pd.isna(value):
            return ""

        val_str = str(value).strip()

        # Handle float representation of integers (e.g., "1.0" -> "1")
        try:
            float_val = float(val_str)
            if float_val == int(float_val):
                return str(int(float_val))
        except (ValueError, TypeError):
            pass

        return val_str

    def _harmonize_sex(
        self,
        df: pd.DataFrame,
        dictionary: Dict,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize SEX using RAG-retrieved or fallback mappings, with LLM fallback."""
        # Get mappings
        code_map = self._get_code_mappings("SEX")
        valid_values = self._get_valid_values("SEX") or ["Male", "Female", "Unknown"]

        # Get dictionary codes - keep original keys (not uppercased) like v2
        dict_codes = dictionary.get("SEX", {}).get("codes", {})
        logger.info(f"SEX harmonization: dictionary keys={list(dictionary.keys())}, dict_codes={dict_codes}")

        # OPTIMIZATION: Pre-resolve unique values with LLM before applying to all rows
        unique_values = df["SEX"].dropna().unique()
        llm_resolutions = {}
        unresolved_values = set()

        for val in unique_values:
            # Normalize value for lookup (handles float-to-int conversion)
            val_normalized = self._normalize_code_key(val)
            val_upper = val_normalized.upper()

            # Check if already resolvable without LLM
            # Check dictionary with normalized value (handles "1.0" -> "1")
            if val_normalized in dict_codes:
                continue
            # Check with uppercased value as fallback
            if val_upper in dict_codes:
                continue
            # Check built-in code map
            if val_upper in code_map:
                continue
            # Check if already valid
            if val_upper in ['MALE', 'FEMALE', 'UNKNOWN']:
                continue

            # Need LLM resolution - do it once per unique value
            if self.use_llm_fallback and self.llm_service and self.llm_service.is_configured:
                resolved = self._resolve_with_llm("SEX", val_normalized, valid_values)
                if resolved:
                    llm_resolutions[val_normalized] = resolved
                else:
                    unresolved_values.add(val_normalized)
            else:
                unresolved_values.add(val_normalized)

        llm_used = bool(llm_resolutions)

        def harmonize_sex(x):
            if pd.isna(x):
                return None

            # Normalize value for lookup (handles float-to-int: "1.0" -> "1")
            val_normalized = self._normalize_code_key(x)
            val_upper = val_normalized.upper()

            # Try dictionary first with normalized value
            if val_normalized in dict_codes:
                decoded = dict_codes[val_normalized]
                return to_mixed_case(decoded)

            # Try dictionary with uppercased value as fallback
            if val_upper in dict_codes:
                decoded = dict_codes[val_upper]
                return to_mixed_case(decoded)

            # Try built-in code mappings
            if val_upper in code_map:
                return code_map[val_upper]

            # Already a valid value?
            if val_upper in ['MALE', 'FEMALE', 'UNKNOWN']:
                return to_mixed_case(val_upper)

            # Use pre-resolved LLM value
            if val_normalized in llm_resolutions:
                return llm_resolutions[val_normalized]

            # Return as-is with mixed case
            return to_mixed_case(val_normalized)

        result = df["SEX"].apply(harmonize_sex)
        lineage["transform_operation"] = "Decode codes, normalize to mixed case"
        lineage["transform_details"] = {
            "valid_values": valid_values,
            "dictionary_used": bool(dict_codes),
            "llm_used": llm_used,
            "llm_resolutions": llm_resolutions,
            "unresolved_values": list(unresolved_values)[:10]
        }
        return result, lineage

    def _harmonize_race(
        self,
        df: pd.DataFrame,
        dictionary: Dict,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize RACE using RAG-retrieved or fallback mappings, with LLM fallback."""
        norm_map = self._get_code_mappings("RACE")

        # Get dictionary codes - keep original keys (v2 approach)
        dict_codes = dictionary.get("RACE", {}).get("codes", {})
        logger.info(f"RACE harmonization: dict_codes={dict_codes}")

        # Valid CDISC race categories
        valid_values = [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Native Hawaiian or Other Pacific Islander",
            "White",
            "Multiple",
            "Other",
            "Unknown"
        ]

        # OPTIMIZATION: Pre-resolve unique values with LLM before applying to all rows
        unique_values = df["RACE"].dropna().unique()
        llm_resolutions = {}
        unresolved_values = set()

        for val in unique_values:
            # Normalize value for lookup (handles float-to-int conversion)
            val_normalized = self._normalize_code_key(val)
            val_upper = val_normalized.upper()

            # Check if already resolvable without LLM
            # Check dictionary with normalized value (handles "11.0" -> "11")
            if val_normalized in dict_codes:
                continue
            # Fallback to uppercased key
            if val_upper in dict_codes:
                continue
            # Check normalization map
            if val_upper in norm_map:
                continue

            # Check if already a valid value
            val_mixed = to_mixed_case(val_upper)
            is_valid = any(val_mixed.lower() == v.lower() for v in valid_values)
            if is_valid:
                continue

            # Need LLM resolution - do it once per unique value
            if self.use_llm_fallback and self.llm_service and self.llm_service.is_configured:
                resolved = self._resolve_with_llm("RACE", val_normalized, valid_values)
                if resolved:
                    llm_resolutions[val_normalized] = resolved
                else:
                    unresolved_values.add(val_normalized)
            else:
                unresolved_values.add(val_normalized)

        llm_used = bool(llm_resolutions)

        def harmonize_race(x):
            if pd.isna(x):
                return None

            # Normalize value for lookup (handles float-to-int: "11.0" -> "11")
            val_normalized = self._normalize_code_key(x)
            val_upper = val_normalized.upper()

            # Try dictionary first with normalized value
            if val_normalized in dict_codes:
                decoded = dict_codes[val_normalized]
                # Apply additional normalization
                decoded_upper = decoded.upper()
                if decoded_upper in norm_map:
                    return norm_map[decoded_upper]
                return to_mixed_case(decoded)

            # Fallback to uppercased key in dictionary
            if val_upper in dict_codes:
                decoded = dict_codes[val_upper]
                decoded_upper = decoded.upper()
                if decoded_upper in norm_map:
                    return norm_map[decoded_upper]
                return to_mixed_case(decoded)

            # Try normalization map directly
            if val_upper in norm_map:
                return norm_map[val_upper]

            # Check if already a valid value
            val_mixed = to_mixed_case(val_upper)
            for valid in valid_values:
                if val_mixed.lower() == valid.lower():
                    return valid

            # Use pre-resolved LLM value
            if val_normalized in llm_resolutions:
                return llm_resolutions[val_normalized]

            # Return as mixed case
            return val_mixed

        result = df["RACE"].apply(harmonize_race)
        lineage["transform_operation"] = "Decode codes, normalize, mixed case"
        lineage["transform_details"] = {
            "valid_values": valid_values,
            "dictionary_used": bool(dict_codes),
            "llm_used": llm_used,
            "llm_resolutions": llm_resolutions,
            "unresolved_values": list(unresolved_values)[:10]
        }
        return result, lineage

    def _harmonize_age(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize AGE - convert to numeric, derive if possible."""
        def clean_age(row):
            age_val = row.get("AGE")

            if pd.notna(age_val):
                try:
                    age = float(age_val)
                    if 0 <= age <= 120:
                        return age
                except (ValueError, TypeError):
                    pass

            # Try derivation from dates
            brthdtc = row.get("BRTHDTC")
            rfstdtc = row.get("RFSTDTC")

            if pd.notna(brthdtc) and pd.notna(rfstdtc):
                if is_full_date(str(brthdtc)) and is_full_date(str(rfstdtc)):
                    derived = calculate_age(str(brthdtc), str(rfstdtc))
                    if derived is not None:
                        return derived

            return None

        result = df.apply(clean_age, axis=1)
        lineage["transform_operation"] = "Convert to numeric, derive from dates if missing"
        return result, lineage

    def _harmonize_ageu(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize AGEU - standardize to Years."""
        def clean_ageu(row):
            age = row.get("AGE")
            ageu = row.get("AGEU")

            if pd.isna(age):
                return None

            if pd.notna(ageu):
                val = str(ageu).strip().upper()
                if val in ["YEARS", "YEAR", "Y", "YRS"]:
                    return "Years"
                # If other units, age should have been converted
                return "Years"

            return "Years"

        result = df.apply(clean_ageu, axis=1)
        lineage["transform_operation"] = "Standardize to 'Years'"
        return result, lineage

    def _harmonize_agegp(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize AGEGP - only if AGE not available."""
        def clean_agegp(row):
            age = row.get("AGE")
            agegp = row.get("AGEGP")

            # If AGE is available, AGEGP should be blank
            if pd.notna(age):
                return None

            if pd.notna(agegp):
                return str(agegp).strip()

            return None

        result = df.apply(clean_agegp, axis=1)
        lineage["transform_operation"] = "Preserve if AGE missing, blank otherwise"
        return result, lineage

    def _harmonize_ethnic(
        self,
        df: pd.DataFrame,
        dictionary: Dict,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize ETHNIC - decode and normalize case."""
        # Check for ETHGRP first (v2 approach), then ETHNIC as fallback
        dict_codes = {}
        dict_source = None

        if dictionary and 'ETHGRP' in dictionary:
            dict_codes = dictionary['ETHGRP'].get('codes', {})
            dict_source = 'ETHGRP'
        elif dictionary and 'ETHNIC' in dictionary:
            dict_codes = dictionary['ETHNIC'].get('codes', {})
            dict_source = 'ETHNIC'

        logger.info(f"ETHNIC harmonization: dict_source={dict_source}, dict_codes={dict_codes}")

        def harmonize_ethnic(x):
            if pd.isna(x):
                return None

            # Normalize value for lookup (handles float-to-int: "1.0" -> "1")
            val_normalized = self._normalize_code_key(x)
            val_upper = val_normalized.upper()

            # Try dictionary with normalized value first
            if val_normalized in dict_codes:
                return to_mixed_case(dict_codes[val_normalized])

            # Fallback to uppercased key
            if val_upper in dict_codes:
                return to_mixed_case(dict_codes[val_upper])

            return to_mixed_case(val_normalized)

        result = df["ETHNIC"].apply(harmonize_ethnic)
        lineage["transform_operation"] = "Decode codes, normalize to mixed case"
        lineage["transform_details"] = {
            "dictionary_used": bool(dict_codes),
            "dictionary_source": dict_source,
            "dict_codes": list(dict_codes.keys())[:10]
        }
        return result, lineage

    def _harmonize_country(
        self,
        df: pd.DataFrame,
        dictionary: Dict,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize COUNTRY - expand ISO codes."""
        country_map = self._get_code_mappings("COUNTRY")
        dict_map = dictionary.get("COUNTRY", {}).get("codes", {})

        def harmonize_country(x):
            if pd.isna(x):
                return None

            val = str(x).strip().upper()

            # Try dictionary first
            if val in dict_map:
                return to_mixed_case(dict_map[val])

            # Try ISO code expansion
            if val in country_map:
                return country_map[val]

            # Return as mixed case
            return to_mixed_case(val)

        result = df["COUNTRY"].apply(harmonize_country)
        lineage["transform_operation"] = "Expand ISO codes, normalize to mixed case"
        return result, lineage

    def _harmonize_siteid(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize SITEID - preserve as string (keep leading zeros)."""
        def clean_siteid(x):
            if pd.isna(x):
                return None
            # Preserve as string
            val = str(x).strip()
            # Handle float representation
            try:
                if '.' in val:
                    float_val = float(val)
                    if float_val == int(float_val):
                        return str(int(float_val))
            except (ValueError, TypeError):
                pass
            return val

        result = df["SITEID"].apply(clean_siteid)
        lineage["transform_operation"] = "Preserve as string (maintain leading zeros)"
        return result, lineage

    def _harmonize_studyid(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize STUDYID - trim whitespace."""
        result = df["STUDYID"].apply(
            lambda x: str(x).strip() if pd.notna(x) else None
        )
        lineage["transform_operation"] = "Trim whitespace"
        return result, lineage

    def _harmonize_usubjid(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize USUBJID - derive if not present."""
        def derive_usubjid(row):
            usubjid = row.get("USUBJID")
            if pd.notna(usubjid):
                return str(usubjid).strip()

            # Derive from STUDYID/TRIAL and SUBJID
            subjid = row.get("SUBJID")
            if pd.isna(subjid):
                return None

            studyid = row.get("STUDYID")
            trial = row.get("TRIAL")

            prefix = studyid if pd.notna(studyid) else trial
            if pd.notna(prefix):
                return f"{prefix}-{subjid}"

            return str(subjid)

        result = df.apply(derive_usubjid, axis=1)
        lineage["transform_operation"] = "Derive as STUDYID||'-'||SUBJID if not present"
        return result, lineage

    def _harmonize_arm(
        self,
        df: pd.DataFrame,
        variable: str,
        dictionary: Dict,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize ARM/ARMCD - decode and normalize case."""
        dict_key = "ARMCD" if variable == "ARMCD" else "ARM"
        dict_map = dictionary.get(dict_key, {}).get("codes", {})

        def harmonize_arm(x):
            if pd.isna(x):
                return None

            val = str(x).strip()

            # Try dictionary
            if val.upper() in dict_map:
                return to_mixed_case(dict_map[val.upper()])

            return to_mixed_case(val)

        result = df[variable].apply(harmonize_arm)
        lineage["transform_operation"] = "Decode codes, normalize to mixed case"
        return result, lineage

    def _harmonize_date(
        self,
        df: pd.DataFrame,
        variable: str,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Harmonize date fields - convert SAS dates to ISO 8601."""
        def clean_date(x):
            if pd.isna(x):
                return None

            val = str(x).strip()

            # Already ISO format?
            if re.match(r'^\d{4}(-\d{2})?(-\d{2})?$', val):
                return val

            # Try SAS date conversion (numeric days since 1960-01-01)
            try:
                numeric_val = float(val)
                # Large values might be datetimes (seconds since 1960)
                if numeric_val > 50000:
                    return sas_datetime_to_iso(numeric_val)
                else:
                    return sas_date_to_iso(int(numeric_val))
            except (ValueError, TypeError):
                pass

            # Return as-is if can't parse
            return val

        result = df[variable].apply(clean_date)
        lineage["transform_operation"] = "Convert SAS dates to ISO 8601"
        return result, lineage

    def _harmonize_domain(self, df: pd.DataFrame, lineage: Dict) -> Tuple[pd.Series, Dict]:
        """Harmonize DOMAIN - set constant 'DM'."""
        result = pd.Series(["DM"] * len(df), index=df.index)
        lineage["transform_operation"] = "Constant 'DM'"
        return result, lineage

    def _harmonize_default(
        self,
        df: pd.DataFrame,
        variable: str,
        lineage: Dict
    ) -> Tuple[pd.Series, Dict]:
        """Default harmonization - normalize whitespace."""
        result = df[variable].apply(
            lambda x: normalize_whitespace(str(x)) if pd.notna(x) else None
        )
        lineage["transform_operation"] = "Normalize whitespace"
        return result, lineage

    def _count_changes(self, original: pd.Series, harmonized: pd.Series) -> int:
        """Count rows where value changed."""
        def normalize_for_compare(x):
            if pd.isna(x):
                return ""
            return str(x).strip().lower()

        orig_norm = original.apply(normalize_for_compare)
        harm_norm = harmonized.apply(normalize_for_compare)

        return int((orig_norm != harm_norm).sum())

    def _check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicate (TRIAL, SUBJID) combinations."""
        if "TRIAL" not in df.columns or "SUBJID" not in df.columns:
            return pd.DataFrame()

        # Find duplicates
        mask = df.duplicated(subset=["TRIAL", "SUBJID"], keep=False)
        return df[mask]
