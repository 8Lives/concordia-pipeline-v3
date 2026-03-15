"""
Specification Retriever

High-level interface for retrieving harmonization rules from the vector store.
This is the primary API that agents use to get specification context.

Usage:
    retriever = SpecificationRetriever(vector_store)

    # Get rules for a specific variable
    rules = retriever.get_variable_rules("SEX")

    # Get source column synonyms
    synonyms = retriever.get_source_columns("SUBJID")

    # Get valid values
    valid = retriever.get_valid_values("RACE")

    # Get QC rules
    qc_rules = retriever.get_qc_rules("TRIAL")

    # Semantic search for any rule
    results = retriever.search("how to handle missing age values")
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vector_store import VectorStore, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class VariableSpec:
    """Complete specification for a variable."""
    variable: str
    required: str
    source_priority: List[str]
    transformation: str
    validation_qc: str
    valid_values: Optional[List[str]]
    spec_reference: str
    confidence: float  # Retrieval confidence score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "required": self.required,
            "source_priority": self.source_priority,
            "transformation": self.transformation,
            "validation_qc": self.validation_qc,
            "valid_values": self.valid_values,
            "spec_reference": self.spec_reference,
            "confidence": self.confidence
        }


class SpecificationRetriever:
    """
    High-level API for retrieving specification rules.

    Provides typed methods for common retrieval patterns:
        - Variable rules (transformation, validation)
        - Source column mappings
        - Valid value codelists
        - QC check definitions
        - General harmonization rules
    """

    def __init__(self, vector_store: VectorStore, default_n_results: int = 5):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance with indexed specifications
            default_n_results: Default number of results for semantic search
        """
        self.vector_store = vector_store
        self.default_n_results = default_n_results

    def get_variable_rules(self, variable: str) -> Optional[VariableSpec]:
        """
        Get the complete specification for a variable.

        Args:
            variable: Variable name (e.g., "SEX", "RACE", "AGE")

        Returns:
            VariableSpec with all rules, or None if not found
        """
        variable = variable.upper()

        # First try exact metadata match
        results = self.vector_store.query_by_metadata(
            where={"variable": variable, "type": "variable_rule"},
            n_results=1
        )

        if results.ids:
            doc = results.documents[0]
            meta = results.metadatas[0]

            # Parse source_priority from JSON string in metadata
            source_priority = []
            if meta.get("source_priority"):
                try:
                    source_priority = json.loads(meta["source_priority"])
                except:
                    source_priority = []

            # Get valid values from codelist
            valid_values = self._get_codelist_values(variable)

            # Parse transformation and validation from document content
            transformation = ""
            validation_qc = ""

            lines = doc.split("\n")
            for line in lines:
                if line.startswith("Transformation/Derivation:"):
                    transformation = line.replace("Transformation/Derivation:", "").strip()
                elif line.startswith("Validation/QC:"):
                    validation_qc = line.replace("Validation/QC:", "").strip()

            return VariableSpec(
                variable=variable,
                required=meta.get("required", "Unknown"),
                source_priority=source_priority,
                transformation=transformation,
                validation_qc=validation_qc,
                valid_values=valid_values,
                spec_reference=meta.get("spec_reference", ""),
                confidence=1.0  # Exact match
            )

        # Fallback to semantic search
        results = self.vector_store.query(
            query_text=f"harmonization rules for {variable} variable",
            n_results=1,
            where={"type": "variable_rule"}
        )

        if results.ids:
            meta = results.metadatas[0]
            return VariableSpec(
                variable=variable,
                required=meta.get("required", "Unknown"),
                source_priority=json.loads(meta.get("source_priority", "[]")),
                transformation="",
                validation_qc="",
                valid_values=None,
                spec_reference=meta.get("spec_reference", ""),
                confidence=1 - results.distances[0] if results.distances else 0.5
            )

        return None

    def get_source_columns(self, variable: str) -> List[str]:
        """
        Get the list of source column synonyms for a variable.

        Args:
            variable: Target output variable name

        Returns:
            Ordered list of source column candidates
        """
        variable = variable.upper()

        # Try exact match first
        results = self.vector_store.query_by_metadata(
            where={"variable": variable, "type": "variable_rule"},
            n_results=1
        )

        if results.ids and results.metadatas:
            meta = results.metadatas[0]
            if meta.get("source_priority"):
                try:
                    return json.loads(meta["source_priority"])
                except:
                    pass

        # Fallback to semantic search
        results = self.vector_store.query(
            query_text=f"source columns synonyms mappings for {variable}",
            n_results=3
        )

        # Extract source priorities from results
        for i, meta in enumerate(results.metadatas):
            if meta.get("source_priority"):
                try:
                    return json.loads(meta["source_priority"])
                except:
                    continue

        return []

    def get_valid_values(self, variable: str) -> Optional[List[str]]:
        """
        Get the valid value codelist for a variable.

        Args:
            variable: Variable name (e.g., "SEX", "RACE")

        Returns:
            List of valid values, or None if no codelist exists
        """
        return self._get_codelist_values(variable.upper())

    def _get_codelist_values(self, variable: str) -> Optional[List[str]]:
        """Internal method to retrieve codelist values."""
        results = self.vector_store.query_by_metadata(
            where={"variable": variable, "type": "codelist"},
            n_results=1
        )

        if results.ids and results.metadatas:
            meta = results.metadatas[0]
            if meta.get("valid_values"):
                try:
                    return json.loads(meta["valid_values"])
                except:
                    pass

        return None

    def get_qc_rules(self, variable: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get QC rules, optionally filtered by variable.

        Args:
            variable: Optional variable name to filter rules

        Returns:
            List of QC rule definitions
        """
        if variable:
            # Search for QC rules related to this variable
            results = self.vector_store.query(
                query_text=f"QC validation check rules for {variable}",
                n_results=5,
                where={"type": "qc_rule"}
            )
        else:
            # Get all QC rules
            results = self.vector_store.query_by_metadata(
                where={"type": "qc_rule"},
                n_results=50
            )

        rules = []
        for i, doc in enumerate(results.documents):
            meta = results.metadatas[i] if results.metadatas else {}
            rules.append({
                "id": results.ids[i],
                "description": doc,
                "issue_type": meta.get("issue_type"),
                "category": meta.get("category"),
                "spec_reference": meta.get("spec_reference")
            })

        return rules

    def get_transformation_rule(self, variable: str) -> Optional[str]:
        """
        Get the transformation rule for a specific variable.

        Args:
            variable: Variable name

        Returns:
            Transformation rule text, or None if not found
        """
        spec = self.get_variable_rules(variable)
        if spec:
            return spec.transformation
        return None

    def get_general_rules(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get general harmonization rules.

        Args:
            category: Optional category filter (e.g., "text_normalization", "date_normalization")

        Returns:
            List of general rule definitions
        """
        where = {"type": "general_rule"}
        if category:
            where["category"] = category

        results = self.vector_store.query_by_metadata(
            where=where,
            n_results=20
        )

        rules = []
        for i, doc in enumerate(results.documents):
            meta = results.metadatas[i] if results.metadatas else {}
            rules.append({
                "id": results.ids[i],
                "content": doc,
                "category": meta.get("category"),
                "spec_reference": meta.get("spec_reference")
            })

        return rules

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across all specification content.

        Args:
            query: Natural language query
            n_results: Number of results (default: self.default_n_results)
            filter_type: Optional type filter (e.g., "variable_rule", "qc_rule", "codelist")

        Returns:
            List of matching specification chunks with scores
        """
        where = None
        if filter_type:
            where = {"type": filter_type}

        results = self.vector_store.query(
            query_text=query,
            n_results=n_results or self.default_n_results,
            where=where
        )

        return results.to_list()

    def get_context_for_llm(
        self,
        variable: Optional[str] = None,
        include_general_rules: bool = True,
        include_qc_rules: bool = True,
        max_tokens: int = 2000
    ) -> str:
        """
        Build context string for LLM prompts.

        Combines relevant specification chunks into a formatted string
        suitable for injection into LLM prompts.

        Args:
            variable: Optional variable to focus on
            include_general_rules: Whether to include general rules
            include_qc_rules: Whether to include QC rules
            max_tokens: Approximate max length (in tokens, ~4 chars each)

        Returns:
            Formatted context string
        """
        sections = []
        char_limit = max_tokens * 4  # Rough estimate

        # Variable-specific rules
        if variable:
            spec = self.get_variable_rules(variable)
            if spec:
                sections.append(f"""## {variable} Variable Rules
- Required: {spec.required}
- Source Priority: {', '.join(spec.source_priority) if spec.source_priority else 'N/A'}
- Transformation: {spec.transformation}
- Validation: {spec.validation_qc}
- Valid Values: {', '.join(spec.valid_values) if spec.valid_values else 'N/A'}
- Reference: {spec.spec_reference}""")

        # General rules
        if include_general_rules:
            general = self.get_general_rules()
            if general:
                rules_text = "\n".join([f"- {r['content']}" for r in general[:5]])
                sections.append(f"## General Harmonization Rules\n{rules_text}")

        # QC rules
        if include_qc_rules:
            qc = self.get_qc_rules(variable)
            if qc:
                qc_text = "\n".join([f"- {r['description']}" for r in qc[:5]])
                sections.append(f"## QC Rules\n{qc_text}")

        context = "\n\n".join(sections)

        # Truncate if needed
        if len(context) > char_limit:
            context = context[:char_limit] + "\n[...truncated]"

        return context

    def get_all_variables(self) -> List[str]:
        """Get list of all variables with indexed rules."""
        results = self.vector_store.query_by_metadata(
            where={"type": "variable_rule"},
            n_results=100
        )

        variables = set()
        for meta in results.metadatas:
            if meta.get("variable"):
                variables.add(meta["variable"])

        return sorted(list(variables))

    def get_output_schema(self) -> List[str]:
        """Get the output schema (ordered list of output variables)."""
        # Search for schema definition
        results = self.vector_store.query(
            query_text="output schema variable order list",
            n_results=1,
            where={"category": "schema"}
        )

        if results.documents:
            # Parse schema from content
            content = results.documents[0]
            if "Output Schema:" in content:
                schema_line = content.split("Output Schema:")[1].split("\n")[0]
                variables = [
                    v.strip().rstrip(".")
                    for v in schema_line.split(",")
                ]
                return variables

        # Fallback to hardcoded order
        return [
            "TRIAL", "SUBJID", "SEX", "RACE", "AGE", "AGEU", "AGEGP", "ETHNIC",
            "COUNTRY", "SITEID", "STUDYID", "USUBJID", "ARMCD", "ARM",
            "BRTHDTC", "RFSTDTC", "RFENDTC", "DOMAIN"
        ]

    def get_required_variables(self) -> List[str]:
        """Get list of required variables."""
        results = self.vector_store.query_by_metadata(
            where={"type": "variable_rule", "required": "Yes"},
            n_results=20
        )

        return [meta.get("variable") for meta in results.metadatas if meta.get("variable")]
