"""
Ingest & Convert Agent (v2.1 - Agentic Architecture)

Responsibilities:
- Accept SAS7BDAT, XPT, CSV, or XLSX files
- Parse file format and extract metadata
- Decode byte strings
- Extract TRIAL ID from filename
- Output standardized DataFrame + metadata

Changes from v2:
- Added .docx and .pdf support for data dictionaries
- Improved dictionary parsing with multi-format support

Changes from v1:
- Extends AgentBase for timeout/retry/callback support
- Uses PipelineContext for input/output
- Standardized error handling
"""
import os
import re
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from agents.base import AgentBase, AgentResult, AgentConfig, PipelineContext, ProgressCallback
from utils.helpers import extract_trial_from_filename, decode_dataframe_bytes


class IngestAgent(AgentBase):
    """
    Agent responsible for ingesting and converting source data files.
    """

    SUPPORTED_FORMATS = ['.sas7bdat', '.xpt', '.csv', '.xlsx', '.xls']

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(
            name="Ingest & Convert Agent",
            config=config or AgentConfig(timeout_seconds=60.0),
            progress_callback=progress_callback
        )

    def validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate required inputs exist."""
        data_file = context.get("data_file")
        if not data_file:
            return "Missing required input: data_file"

        if not os.path.exists(data_file):
            return f"File not found: {data_file}"

        file_ext = Path(data_file).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return f"Unsupported format: {file_ext}. Supported: {self.SUPPORTED_FORMATS}"

        return None

    def execute(self, context: PipelineContext) -> AgentResult:
        """
        Execute file ingestion.

        Reads from context:
        - data_file: Path to source file
        - dictionary_file: Optional path to dictionary

        Writes to context (via result.data):
        - df: Parsed DataFrame
        - ingest_metadata: Metadata about the file
        """
        file_path = Path(context.get("data_file"))
        dictionary_path = context.get("dictionary_file")
        file_ext = file_path.suffix.lower()

        # Update progress
        self._update_status(self.status, "Parsing source file...", 0.2)

        # Extract TRIAL from filename
        trial_id = extract_trial_from_filename(file_path.name)

        # Parse file
        df, parse_metadata = self._parse_file(file_path, file_ext)

        if df is None:
            return AgentResult(
                success=False,
                error=parse_metadata.get('error', 'Unknown parsing error'),
                error_type="ParseError"
            )

        self._update_status(self.status, "Decoding byte strings...", 0.5)

        # Decode byte strings
        df = decode_dataframe_bytes(df)

        # Parse dictionary if provided
        self._update_status(self.status, "Loading dictionary...", 0.7)
        dictionary = None
        dict_metadata = {}
        if dictionary_path and os.path.exists(dictionary_path):
            dictionary, dict_metadata = self._parse_dictionary(dictionary_path)

        self._update_status(self.status, "Building metadata...", 0.9)

        # Build metadata
        metadata = {
            "agent": self.name,
            "source_file": str(file_path),
            "source_filename": file_path.name,
            "file_format": file_ext,
            "trial_id": trial_id,  # Simplified key name
            "trial_id_from_filename": trial_id,  # Keep for backward compat
            "rows": len(df),
            "rows_in": len(df),  # Keep for backward compat
            "columns_in": len(df.columns),
            "column_names": list(df.columns),
            "dictionary_file": dictionary_path,
            "dictionary_loaded": dictionary is not None,
            **parse_metadata,
            **dict_metadata
        }

        # Attach dictionary to metadata if loaded
        if dictionary is not None:
            metadata["dictionary"] = dictionary

        return AgentResult(
            success=True,
            data={
                "df": df,
                "source_df": df.copy(),  # Preserve original for appending unused columns later
                "ingest_metadata": metadata,
                # Flatten key items for easy access
                "trial_id": trial_id,
                "column_names": list(df.columns),
                "dictionary": dictionary
            },
            metadata=metadata
        )

    def _parse_file(self, file_path: Path, file_ext: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Parse file based on extension."""
        metadata = {}

        try:
            if file_ext == '.sas7bdat':
                df = pd.read_sas(file_path, format='sas7bdat')
                metadata['sas_format'] = 'sas7bdat'

            elif file_ext == '.xpt':
                df = pd.read_sas(file_path, format='xport')
                metadata['sas_format'] = 'xport'

            elif file_ext == '.csv':
                df = pd.read_csv(file_path, dtype=str)
                metadata['csv_encoding'] = 'utf-8'

            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, dtype=str)
                metadata['excel_format'] = file_ext

            else:
                return None, {'error': f'Unsupported format: {file_ext}'}

            return df, metadata

        except Exception as e:
            return None, {'error': str(e)}

    def _parse_dictionary(self, dict_path: str) -> Tuple[Optional[Dict], Dict[str, Any]]:
        """Parse a data dictionary file. Supports xlsx, xls, csv, docx, and pdf."""
        try:
            dict_path = Path(dict_path)
            file_ext = dict_path.suffix.lower()

            if file_ext in ['.xlsx', '.xls']:
                xlsx = pd.ExcelFile(dict_path)
                dictionary = {}
                active_sheet = None

                for sheet_name in xlsx.sheet_names:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)
                    dict_data = self._extract_dictionary_from_sheet(df, sheet_name)
                    if dict_data:
                        dictionary.update(dict_data)
                        if active_sheet is None:
                            active_sheet = sheet_name

                return dictionary if dictionary else None, {
                    "dictionary_sheets": xlsx.sheet_names,
                    "dictionary_filename": dict_path.name,
                    "dictionary_active_sheet": active_sheet,
                    "dictionary_format": "excel"
                }

            elif file_ext == '.csv':
                df = pd.read_csv(dict_path)
                dictionary = self._extract_dictionary_from_sheet(df, 'main')
                return dictionary, {"dictionary_format": "csv", "dictionary_filename": dict_path.name}

            elif file_ext == '.docx':
                dictionary, metadata = self._parse_docx_dictionary(dict_path)
                return dictionary, metadata

            elif file_ext == '.pdf':
                dictionary, metadata = self._parse_pdf_dictionary(dict_path)
                return dictionary, metadata

            return None, {"dictionary_error": f"Unsupported dictionary format: {file_ext}"}

        except Exception as e:
            return None, {"dictionary_error": str(e)}

    def _parse_docx_dictionary(self, dict_path: Path) -> Tuple[Optional[Dict], Dict[str, Any]]:
        """
        Parse a data dictionary from a .docx file.
        Uses pandoc to convert to markdown, then extracts code mappings.
        """
        metadata = {
            "dictionary_format": "docx",
            "dictionary_filename": dict_path.name
        }

        try:
            # Use pandoc to convert docx to plain text/markdown
            result = subprocess.run(
                ['pandoc', str(dict_path), '-t', 'plain'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                metadata["dictionary_error"] = f"Pandoc conversion failed: {result.stderr}"
                return None, metadata

            text_content = result.stdout
            dictionary = self._extract_dictionary_from_text(text_content)

            if dictionary:
                metadata["dictionary_variables_found"] = list(dictionary.keys())
                return dictionary, metadata
            else:
                metadata["dictionary_warning"] = "No code mappings found in document"
                return None, metadata

        except subprocess.TimeoutExpired:
            metadata["dictionary_error"] = "Pandoc conversion timed out"
            return None, metadata
        except FileNotFoundError:
            metadata["dictionary_error"] = "Pandoc not installed - cannot parse .docx dictionaries"
            return None, metadata
        except Exception as e:
            metadata["dictionary_error"] = str(e)
            return None, metadata

    def _parse_pdf_dictionary(self, dict_path: Path) -> Tuple[Optional[Dict], Dict[str, Any]]:
        """
        Parse a data dictionary from a .pdf file.
        Uses pdftotext (from poppler) to extract text, then extracts code mappings.
        """
        metadata = {
            "dictionary_format": "pdf",
            "dictionary_filename": dict_path.name
        }

        try:
            # Try pdftotext first (better for tabular content)
            result = subprocess.run(
                ['pdftotext', '-layout', str(dict_path), '-'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                # Fallback to pandoc
                result = subprocess.run(
                    ['pandoc', str(dict_path), '-t', 'plain'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            if result.returncode != 0:
                metadata["dictionary_error"] = f"PDF extraction failed: {result.stderr}"
                return None, metadata

            text_content = result.stdout
            dictionary = self._extract_dictionary_from_text(text_content)

            if dictionary:
                metadata["dictionary_variables_found"] = list(dictionary.keys())
                return dictionary, metadata
            else:
                metadata["dictionary_warning"] = "No code mappings found in PDF"
                return None, metadata

        except subprocess.TimeoutExpired:
            metadata["dictionary_error"] = "PDF extraction timed out"
            return None, metadata
        except FileNotFoundError:
            metadata["dictionary_error"] = "pdftotext/pandoc not installed - cannot parse .pdf dictionaries"
            return None, metadata
        except Exception as e:
            metadata["dictionary_error"] = str(e)
            return None, metadata

    def _extract_dictionary_from_text(self, text: str) -> Dict:
        """
        Extract variable code mappings from plain text.
        Looks for patterns like:
        - "1 = Male" or "1=Male"
        - "VARIABLE: SEX" followed by code mappings
        - Table-like structures with codes and labels
        """
        dictionary = {}
        current_var = None

        # Common variable names in clinical data
        known_vars = ['SEX', 'RACE', 'ETHNIC', 'ETHGRP', 'COUNTRY', 'ARMCD', 'ARM',
                      'AGEU', 'AGEUNITS', 'SITEID', 'TRTCODE', 'TRT']

        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line contains a variable name
            line_upper = line.upper()
            for var in known_vars:
                # Match patterns like "SEX:" or "Variable: SEX" or just "SEX" at start
                if re.match(rf'^{var}\s*[:=]?\s*$', line_upper) or \
                   re.match(rf'^(VARIABLE|VAR|FIELD)\s*[:=]?\s*{var}', line_upper, re.IGNORECASE) or \
                   (line_upper == var):
                    current_var = var
                    if current_var not in dictionary:
                        dictionary[current_var] = {"codes": {}, "format": ""}
                    break

            # Look for code = value patterns
            # Matches: "1 = Male", "1=Male", "1 - Male", "01 = Male"
            code_match = re.match(r'^(\d+)\s*[=\-:]\s*(.+)$', line)
            if code_match and current_var:
                code = code_match.group(1).strip()
                label = code_match.group(2).strip()
                # Clean up label (remove trailing punctuation, etc.)
                label = re.sub(r'[,;]$', '', label).strip()
                if label and len(label) < 100:  # Sanity check
                    dictionary[current_var]["codes"][code] = label

            # Also check for letter codes like "M = Male", "F = Female"
            letter_match = re.match(r'^([A-Z])\s*[=\-:]\s*(.+)$', line, re.IGNORECASE)
            if letter_match and current_var:
                code = letter_match.group(1).strip().upper()
                label = letter_match.group(2).strip()
                label = re.sub(r'[,;]$', '', label).strip()
                if label and len(label) < 100:
                    dictionary[current_var]["codes"][code] = label

        # Remove empty entries
        dictionary = {k: v for k, v in dictionary.items() if v.get("codes")}

        return dictionary

    def _extract_dictionary_from_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """Extract variable code mappings from a dictionary sheet."""
        dictionary = {}

        # Find the header row
        header_row_idx = None
        for idx, row in df.iterrows():
            for col_idx, cell in enumerate(row):
                if pd.notna(cell):
                    cell_str = str(cell).strip().upper()
                    if cell_str in ['VARIABLE NAME', 'VARIABLE', 'VAR NAME']:
                        header_row_idx = idx
                        break
            if header_row_idx is not None:
                break

        if header_row_idx is not None:
            new_columns = df.iloc[header_row_idx].tolist()
            df = df.iloc[header_row_idx + 1:].copy()
            df.columns = [str(c).strip() if pd.notna(c) else f'col_{i}' for i, c in enumerate(new_columns)]

        # Identify key columns
        col_map = {str(c).upper().replace('\n', ' '): c for c in df.columns}

        var_col = None
        value_col = None
        format_col = None

        for candidate in ['VARIABLE NAME', 'VARIABLE', 'VAR', 'NAME', 'FIELD']:
            if candidate in col_map:
                var_col = col_map[candidate]
                break

        for candidate in ['VALID VALUES', 'VALUES', 'DECODE', 'VALID VALUE']:
            if candidate in col_map:
                value_col = col_map[candidate]
                break

        for candidate in ['FORMAT  (VALUE LIST)', 'FORMAT (VALUE LIST)', 'FORMAT', 'VALUE LIST', 'CODELIST']:
            if candidate in col_map:
                format_col = col_map[candidate]
                break

        if not var_col:
            return {}

        current_var = None

        for _, row in df.iterrows():
            var_name = row.get(var_col)

            if pd.notna(var_name) and str(var_name).strip():
                current_var = str(var_name).strip().upper()
                if current_var not in dictionary:
                    dictionary[current_var] = {
                        "codes": {},
                        "format": str(row.get(format_col, '')) if format_col and pd.notna(row.get(format_col)) else ''
                    }

            if current_var and value_col:
                value_str = row.get(value_col)
                if pd.notna(value_str):
                    value_str = str(value_str).strip()
                    if '=' in value_str:
                        parts = value_str.split('=', 1)
                        code = parts[0].strip()
                        label = parts[1].strip()
                        dictionary[current_var]["codes"][code] = label

        dictionary = {k: v for k, v in dictionary.items() if v.get("codes")}

        return dictionary
