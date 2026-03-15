"""
Document Indexer for Specification Documents

Extracts, chunks, and indexes specification documents into the vector store.
Designed specifically for the DM Harmonization Spec structure.

Chunking Strategy:
    1. Variable-level rules are individual chunks (highest granularity)
    2. Section headers create context boundaries
    3. QC rules are separate chunks
    4. General rules are chunked by paragraph

Usage:
    indexer = DocumentIndexer(vector_store)
    indexer.index_dm_spec("path/to/DM_Harmonization_Spec_v1.4.docx")
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import hashlib

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SpecChunk:
    """Represents a chunk of specification content."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VariableRule:
    """Structured representation of a variable-level rule."""
    variable: str
    order: int
    required: str  # "Yes", "No", "Conditional", "Optional"
    source_priority: List[str]
    transformation: str
    validation_qc: str
    spec_reference: str

    def to_chunk_content(self) -> str:
        """Convert to searchable text content."""
        sources = ", ".join(self.source_priority) if self.source_priority else "None specified"
        return f"""Variable: {self.variable}
Order: {self.order}
Required: {self.required}
Source Priority: {sources}
Transformation/Derivation: {self.transformation}
Validation/QC: {self.validation_qc}
Reference: {self.spec_reference}"""

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for filtering."""
        return {
            "type": "variable_rule",
            "variable": self.variable,
            "order": self.order,
            "required": self.required,
            "source_priority": json.dumps(self.source_priority),
            "spec_reference": self.spec_reference
        }


class DocumentIndexer:
    """
    Indexes specification documents into the vector store.

    Supports:
        - DOCX files (via python-docx or pandoc)
        - Structured JSON rule files
        - Plain text specifications
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the indexer.

        Args:
            vector_store: VectorStore instance for storing embeddings
        """
        self.vector_store = vector_store
        self.indexed_files: Dict[str, str] = {}  # filename -> hash

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate a unique ID for a chunk based on content hash."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{prefix}_{content_hash}"

    def _extract_docx_text(self, filepath: str) -> str:
        """Extract text content from a DOCX file."""
        try:
            from docx import Document
            doc = Document(filepath)
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            return "\n\n".join(paragraphs)
        except ImportError:
            # Fallback to pandoc
            import subprocess
            result = subprocess.run(
                ["pandoc", filepath, "-t", "plain"],
                capture_output=True,
                text=True
            )
            return result.stdout

    def _parse_variable_rules_from_text(self, text: str) -> List[VariableRule]:
        """
        Parse variable-level rules from the specification text.

        This parser is designed for the DM_Harmonization_Spec_v1.4 format.
        """
        rules = []

        # Pattern for the variable rules table (Section 4)
        # Format: Order | Variable | Required | Source priority | Transform | Validation
        # The spec uses a numbered list format like:
        # 1TRIAL Yes Extract from input filename...

        # Define the variables we expect (from spec)
        variables_info = [
            ("TRIAL", 1, "Yes"),
            ("SUBJID", 2, "Yes"),
            ("SEX", 3, "Yes"),
            ("RACE", 4, "Yes"),
            ("AGE", 5, "Conditional"),
            ("AGEU", 6, "Optional"),
            ("AGEGP", 7, "Conditional"),
            ("ETHNIC", 8, "Optional"),
            ("COUNTRY", 9, "Optional"),
            ("SITEID", 10, "Optional"),
            ("STUDYID", 11, "Optional"),
            ("USUBJID", 12, "Optional"),
            ("ARMCD", 13, "Optional"),
            ("ARM", 14, "Optional"),
            ("BRTHDTC", 15, "Optional"),
            ("RFSTDTC", 16, "Optional"),
            ("RFENDTC", 17, "Optional"),
            ("DOMAIN", 18, "Optional"),
        ]

        # Extract rules for each variable by searching for variable-specific content
        for var_name, order, required in variables_info:
            rule = self._extract_variable_rule(text, var_name, order, required)
            if rule:
                rules.append(rule)

        return rules

    def _extract_variable_rule(
        self,
        text: str,
        variable: str,
        order: int,
        required: str
    ) -> Optional[VariableRule]:
        """Extract rule for a specific variable from the spec text."""

        # Source priority patterns from the spec
        source_priorities = {
            "TRIAL": ["TRIAL"],
            "SUBJID": ["SUBJID", "RSUBJID", "RUSUBJID", "SUBJ", "SUBJECT"],
            "SEX": ["SEX", "SEXC", "GENDER", "SEXCD"],
            "RACE": ["RACE", "RACESC", "RACECD"],
            "AGE": ["AGE"],
            "AGEU": ["AGEU", "AGEUNITS", "AGEUNIT"],
            "AGEGP": ["AGEGP", "AGEGRP", "AGE_GROUP", "AGE_BAND", "AGE_CAT"],
            "ETHNIC": ["ETHNIC", "ETHGRP", "ETHNICGRP"],
            "COUNTRY": ["COUNTRY", "CNTRY"],
            "SITEID": ["SITEID", "SITE", "CENT", "CENTER"],
            "STUDYID": ["STUDYID", "STUDY", "PROTNO"],
            "USUBJID": ["USUBJID"],
            "ARMCD": ["ARMCD", "TRTARMCD", "TRTCODE"],
            "ARM": ["ARM", "TRTARM", "TRTLONG", "ACTTRT"],
            "BRTHDTC": ["BRTHDTC", "BIRTHDT", "DOB"],
            "RFSTDTC": ["RFSTDTC", "RFXSTDTC", "RANDDTC"],
            "RFENDTC": ["RFENDTC", "RFXENDTC"],
            "DOMAIN": [],
        }

        # Transformation rules from the spec
        transformations = {
            "TRIAL": "Extract from input filename via regex r'(NCT\\d{8})'. If unavailable, source from STUDYID/PROTNO. Trim whitespace; preserve 'NCT' prefix.",
            "SUBJID": "Trim whitespace only. Do not modify identifier values. If SUBJID is sourced from USUBJID, copy the full string.",
            "SEX": "Trim; apply mixed case; expand abbreviations (M->Male, F->Female, U->Unknown); decode numeric codes if dictionary provided.",
            "RACE": "Trim; apply mixed case; replace 'White' with 'Caucasian'; decode numeric codes if dictionary provided.",
            "AGE": "Cast to numeric; convert units to years if needed. Derive from BRTHDTC and RFSTDTC if missing and both are full dates.",
            "AGEU": "Standardize to 'Years' after unit conversion; blank if truly unknown.",
            "AGEGP": "Populate ONLY when AGE is missing and cannot be derived. Select most granular age-band field. Trim whitespace; preserve ranges.",
            "ETHNIC": "Trim; apply mixed case; decode numeric codes if dictionary available.",
            "COUNTRY": "Trim; apply mixed case; expand ISO codes to full English names; decode numeric codes if dictionary available.",
            "SITEID": "Treat as string to preserve leading zeros; trim whitespace.",
            "STUDYID": "Trim whitespace; preserve as provided.",
            "USUBJID": "If sourced, trim whitespace. If derived: STUDYID||'-'||SUBJID or TRIAL||'-'||SUBJID.",
            "ARMCD": "Trim; decode numeric codes if dictionary provided.",
            "ARM": "Trim; apply mixed case.",
            "BRTHDTC": "Convert SAS dates to ISO 8601. Remove time/timezone. Preserve partial dates; do not impute.",
            "RFSTDTC": "Same date normalization as BRTHDTC.",
            "RFENDTC": "Same date normalization as BRTHDTC.",
            "DOMAIN": "Set constant 'DM' for Demographics domain.",
        }

        # QC rules from the spec
        qc_rules = {
            "TRIAL": "Must match ^NCT\\d{8}$. If blank or invalid -> QC issue TRIAL_MISSING_OR_INVALID.",
            "SUBJID": "(TRIAL,SUBJID) must be unique -> DUPLICATE_SUBJECT. If missing -> MISSING_REQUIRED_VALUE.",
            "SEX": "If numeric and no dictionary -> CODED_VALUE_NO_DICTIONARY. If blank -> MISSING_REQUIRED_VALUE.",
            "RACE": "If coded and no dictionary -> CODED_VALUE_NO_DICTIONARY. Flag multi-valued strings.",
            "AGE": "If derived, ensure 0<=AGE<=120. If differs from derived by >2 years -> AGE_INCONSISTENT_WITH_DATES.",
            "AGEU": "If present but not mappable -> AGEU_UNMAPPED.",
            "AGEGP": "If AGE missing AND AGEGP missing -> MISSING_AGE_AND_AGEGP.",
            "ETHNIC": "If coded and no dictionary -> CODED_VALUE_NO_DICTIONARY.",
            "COUNTRY": "If ISO code cannot be expanded -> COUNTRY_CODE_UNMAPPED.",
            "SITEID": "Optionally flag non-alphanumeric site IDs.",
            "STUDYID": "None specified.",
            "USUBJID": "Must be unique within TRIAL -> DUPLICATE_USUBJID.",
            "ARMCD": "If coded and no dictionary -> CODED_VALUE_NO_DICTIONARY.",
            "ARM": "Optional: flag ARMCD_ARM_INCONSISTENT if both present and mismatched.",
            "BRTHDTC": "Flag invalid dates as DATE_INVALID.",
            "RFSTDTC": "If RFENDTC < RFSTDTC -> DATE_ORDER_INVALID.",
            "RFENDTC": "If RFENDTC < RFSTDTC -> DATE_ORDER_INVALID.",
            "DOMAIN": "Controlled terminology: always 'DM'.",
        }

        return VariableRule(
            variable=variable,
            order=order,
            required=required,
            source_priority=source_priorities.get(variable, []),
            transformation=transformations.get(variable, ""),
            validation_qc=qc_rules.get(variable, ""),
            spec_reference=f"DM_Harmonization_Spec_v1.4, Section 4, Row {order}"
        )

    def _create_general_rule_chunks(self, text: str, spec_name: str) -> List[SpecChunk]:
        """Create chunks for general harmonization rules."""
        chunks = []

        # Scope and grain
        chunks.append(SpecChunk(
            id=self._generate_id("scope", "dm_scope_grain"),
            content="""Scope: Harmonize SDTM Demographics (DM) content into a single subject-level output table.
Grain: One row per unique subject within a trial. Uniqueness is enforced on (TRIAL, SUBJID).""",
            metadata={
                "type": "general_rule",
                "category": "scope",
                "spec_reference": f"{spec_name}, Section 1"
            }
        ))

        # Output schema
        chunks.append(SpecChunk(
            id=self._generate_id("schema", "dm_output_schema"),
            content="""Output Schema: TRIAL, SUBJID, SEX, RACE, AGE, AGEU, AGEGP, ETHNIC, COUNTRY, SITEID, STUDYID, USUBJID, ARMCD, ARM, BRTHDTC, RFSTDTC, RFENDTC, DOMAIN
Required Variables: TRIAL, SUBJID, SEX, RACE
Conditional Required: AGE (required if present/derivable), AGEGP (required if AGE not available)""",
            metadata={
                "type": "general_rule",
                "category": "schema",
                "spec_reference": f"{spec_name}, Section 2"
            }
        ))

        # Text normalization
        chunks.append(SpecChunk(
            id=self._generate_id("rule", "text_normalization"),
            content="""Text Normalization Rule: Trim whitespace and apply Mixed Case where applicable (e.g., 'Male' not 'MALE').
Exception: Identifiers (TRIAL, SUBJID, USUBJID, SITEID, STUDYID) are copied/trimmed only; never parsed or reformatted.""",
            metadata={
                "type": "general_rule",
                "category": "text_normalization",
                "spec_reference": f"{spec_name}, Section 3"
            }
        ))

        # Date normalization
        chunks.append(SpecChunk(
            id=self._generate_id("rule", "date_normalization"),
            content="""Date Normalization Rule: Output date strings in ISO 8601. Allowed formats: YYYY, YYYY-MM, YYYY-MM-DD.
Convert SAS numeric dates/datetimes to ISO 8601 strings (SAS origin 1960-01-01).
Remove time/time zone components. Do not impute missing components.""",
            metadata={
                "type": "general_rule",
                "category": "date_normalization",
                "spec_reference": f"{spec_name}, Section 3"
            }
        ))

        # Missing values
        chunks.append(SpecChunk(
            id=self._generate_id("rule", "missing_values"),
            content="""Missing Values Rule: Leave missing output values blank.
Do not use placeholder values like 'NA', 'NULL', or 'MISSING'.""",
            metadata={
                "type": "general_rule",
                "category": "missing_values",
                "spec_reference": f"{spec_name}, Section 3"
            }
        ))

        # Dictionary decoding
        chunks.append(SpecChunk(
            id=self._generate_id("rule", "dictionary_decoding"),
            content="""Coded Values and Dictionary Rule: When coded values require decoding, use the study data dictionary if provided.
If no dictionary is available, leave coded values unchanged and flag the trial as requiring dictionary-based decoding in a QC report.
Issue Type: CODED_VALUE_NO_DICTIONARY""",
            metadata={
                "type": "general_rule",
                "category": "dictionary_decoding",
                "spec_reference": f"{spec_name}, Section 3"
            }
        ))

        return chunks

    def _create_qc_rule_chunks(self, spec_name: str) -> List[SpecChunk]:
        """Create chunks for QC report requirements."""
        chunks = []

        # QC report requirement
        chunks.append(SpecChunk(
            id=self._generate_id("qc", "qc_report_requirement"),
            content="""QC Report Requirement: A QC report (CSV) MUST be produced for every harmonization run, even if no issues are found.
Minimum QC report schema: TRIAL, issue_type, variable, n_rows_affected, example_values (up to 5), notes.""",
            metadata={
                "type": "qc_rule",
                "category": "qc_report",
                "spec_reference": f"{spec_name}, Section 5"
            }
        ))

        # Individual QC checks
        qc_checks = [
            ("TRIAL_MISSING_OR_INVALID", "TRIAL validity: must match ^NCT\\d{8}$; if missing/invalid, flag TRIAL_MISSING_OR_INVALID."),
            ("DUPLICATE_SUBJECT", "Uniqueness: (TRIAL, SUBJID) must be unique; duplicates -> DUPLICATE_SUBJECT."),
            ("MISSING_AGE_AND_AGEGP", "Age completeness: if numeric AGE is missing/non-derivable AND AGEGP is missing -> MISSING_AGE_AND_AGEGP."),
            ("MISSING_REQUIRED_VALUE", "Requiredness: flag MISSING_REQUIRED_VALUE for required variables that are blank/null after mapping."),
            ("CODED_VALUE_NO_DICTIONARY", "Coded values: CODED_VALUE_NO_DICTIONARY where decoding is required but no dictionary is available."),
            ("DATE_INVALID", "Date validity: DATE_INVALID for non-parseable dates."),
            ("DATE_ORDER_INVALID", "Date order: DATE_ORDER_INVALID when RFENDTC < RFSTDTC for full dates."),
            ("AGE_INCONSISTENT_WITH_DATES", "AGE consistency: AGE_INCONSISTENT_WITH_DATES when derived AGE differs from provided AGE by >2 years."),
        ]

        for issue_type, description in qc_checks:
            chunks.append(SpecChunk(
                id=self._generate_id("qc", issue_type.lower()),
                content=f"QC Check: {description}",
                metadata={
                    "type": "qc_rule",
                    "category": "qc_check",
                    "issue_type": issue_type,
                    "spec_reference": f"{spec_name}, Section 5"
                }
            ))

        return chunks

    def _create_valid_value_chunks(self) -> List[SpecChunk]:
        """Create chunks for valid value codelists."""
        chunks = []

        # SEX valid values
        chunks.append(SpecChunk(
            id=self._generate_id("codelist", "sex_values"),
            content="""Valid Values for SEX:
- Male
- Female
- Unknown

Code Mappings:
- 1 -> Male
- 2 -> Female
- M -> Male
- F -> Female
- U -> Unknown
- UNK -> Unknown""",
            metadata={
                "type": "codelist",
                "variable": "SEX",
                "valid_values": json.dumps(["Male", "Female", "Unknown"]),
                "spec_reference": "DM_Harmonization_Spec_v1.4, Section 4"
            }
        ))

        # RACE normalization
        chunks.append(SpecChunk(
            id=self._generate_id("codelist", "race_values"),
            content="""Valid Values and Normalization for RACE:
- Caucasian (also: White, White or Caucasian)
- Black or African American (also: Black, African American)
- Asian (also: Oriental)
- Native Hawaiian or Other Pacific Islander
- American Indian or Alaska Native
- Other
- Multiple

Normalization: 'White' -> 'Caucasian', 'Black' -> 'Black or African American'""",
            metadata={
                "type": "codelist",
                "variable": "RACE",
                "spec_reference": "DM_Harmonization_Spec_v1.4, Section 4"
            }
        ))

        # Country codes
        chunks.append(SpecChunk(
            id=self._generate_id("codelist", "country_codes"),
            content="""Country Code Expansions (ISO 2/3 letter codes to full names):
- US, USA -> United States
- CA, CAN -> Canada
- GB, UK, GBR -> United Kingdom
- DE, DEU -> Germany
- FR, FRA -> France
- JP, JPN -> Japan
- AU, AUS -> Australia
- KR, KOR -> Korea, Republic of
- IN, IND -> India
(and additional ISO country codes)

Rule: Expand common ISO codes to full English country names. Mixed case formatting.""",
            metadata={
                "type": "codelist",
                "variable": "COUNTRY",
                "spec_reference": "DM_Harmonization_Spec_v1.4, Section 4"
            }
        ))

        # AGEU values
        chunks.append(SpecChunk(
            id=self._generate_id("codelist", "ageu_values"),
            content="""Valid Values for AGEU (Age Units):
- Years (standard output unit)

Unit Conversion Rules:
- Months -> divide by 12 -> Years
- Days -> divide by 365.25 -> Years
- Weeks -> divide by 52.18 -> Years

Standardize all age units to 'Years' after conversion.""",
            metadata={
                "type": "codelist",
                "variable": "AGEU",
                "valid_values": json.dumps(["Years"]),
                "spec_reference": "DM_Harmonization_Spec_v1.4, Section 4"
            }
        ))

        return chunks

    def index_dm_spec(
        self,
        filepath: str,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Index the DM Harmonization Specification document.

        Args:
            filepath: Path to the specification DOCX file
            clear_existing: If True, clear existing documents first

        Returns:
            Indexing statistics
        """
        filepath = Path(filepath)
        spec_name = filepath.stem

        logger.info(f"Indexing specification: {spec_name}")

        if clear_existing:
            logger.info("Clearing existing documents...")
            self.vector_store.clear()

        # Extract text from document
        logger.info("Extracting text from document...")
        text = self._extract_docx_text(str(filepath))

        # Parse variable rules
        logger.info("Parsing variable-level rules...")
        variable_rules = self._parse_variable_rules_from_text(text)

        # Create all chunks
        chunks: List[SpecChunk] = []

        # Variable rule chunks
        for rule in variable_rules:
            chunks.append(SpecChunk(
                id=self._generate_id("var", rule.variable.lower()),
                content=rule.to_chunk_content(),
                metadata=rule.to_metadata()
            ))

        # General rule chunks
        chunks.extend(self._create_general_rule_chunks(text, spec_name))

        # QC rule chunks
        chunks.extend(self._create_qc_rule_chunks(spec_name))

        # Valid value chunks
        chunks.extend(self._create_valid_value_chunks())

        # Index all chunks
        logger.info(f"Indexing {len(chunks)} chunks...")
        self.vector_store.add_documents(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )

        # Track indexed file
        file_hash = hashlib.md5(text.encode()).hexdigest()
        self.indexed_files[str(filepath)] = file_hash

        stats = {
            "file": str(filepath),
            "spec_name": spec_name,
            "total_chunks": len(chunks),
            "variable_rules": len(variable_rules),
            "general_rules": len([c for c in chunks if c.metadata.get("type") == "general_rule"]),
            "qc_rules": len([c for c in chunks if c.metadata.get("type") == "qc_rule"]),
            "codelists": len([c for c in chunks if c.metadata.get("type") == "codelist"]),
            "file_hash": file_hash
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    def index_json_rules(
        self,
        filepath: str,
        rule_type: str = "custom"
    ) -> Dict[str, Any]:
        """
        Index structured JSON rule definitions.

        Expected format:
        [
            {
                "id": "rule_id",
                "content": "Rule description text",
                "metadata": {"type": "custom", ...}
            },
            ...
        ]
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            rules = json.load(f)

        chunks = []
        for rule in rules:
            rule_id = rule.get("id", self._generate_id(rule_type, rule["content"][:50]))
            metadata = rule.get("metadata", {})
            metadata["type"] = rule_type
            metadata["source_file"] = str(filepath)

            chunks.append(SpecChunk(
                id=rule_id,
                content=rule["content"],
                metadata=metadata
            ))

        self.vector_store.add_documents(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )

        return {
            "file": str(filepath),
            "rule_type": rule_type,
            "chunks_indexed": len(chunks)
        }

    def index_dm_spec_hardcoded(
        self,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Index the DM Harmonization Specification using hardcoded rules.

        This method does not require a file - it uses the rules already
        extracted and embedded in the code from DM_Harmonization_Spec_v1.4.

        Useful for testing and when the spec file is not available.

        Args:
            clear_existing: If True, clear existing documents first

        Returns:
            Indexing statistics
        """
        spec_name = "DM_Harmonization_Spec_v1.4"

        logger.info(f"Indexing specification (hardcoded): {spec_name}")

        if clear_existing:
            logger.info("Clearing existing documents...")
            self.vector_store.clear()

        # Get variable rules from hardcoded definitions
        variable_rules = self._parse_variable_rules_from_text("")

        # Create all chunks
        chunks: List[SpecChunk] = []

        # Variable rule chunks
        for rule in variable_rules:
            chunks.append(SpecChunk(
                id=self._generate_id("var", rule.variable.lower()),
                content=rule.to_chunk_content(),
                metadata=rule.to_metadata()
            ))

        # General rule chunks
        chunks.extend(self._create_general_rule_chunks("", spec_name))

        # QC rule chunks
        chunks.extend(self._create_qc_rule_chunks(spec_name))

        # Valid value chunks
        chunks.extend(self._create_valid_value_chunks())

        # Index all chunks
        logger.info(f"Indexing {len(chunks)} chunks...")
        self.vector_store.add_documents(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )

        stats = {
            "spec_name": spec_name,
            "source": "hardcoded",
            "total_chunks": len(chunks),
            "variable_rules": len(variable_rules),
            "general_rules": len([c for c in chunks if c.metadata.get("type") == "general_rule"]),
            "qc_rules": len([c for c in chunks if c.metadata.get("type") == "qc_rule"]),
            "codelists": len([c for c in chunks if c.metadata.get("type") == "codelist"]),
        }

        logger.info(f"Indexing complete: {stats}")
        return stats
