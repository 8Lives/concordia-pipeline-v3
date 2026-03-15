"""
Concordia Pipeline v3 - Streamlit Application

RAG-enhanced clinical trial data harmonization with:
- Dynamic specification retrieval
- Real-time progress tracking
- Interactive QC review
- Full traceability

Run with: streamlit run app.py

Secrets Configuration:
    Create .streamlit/secrets.toml with:
        HUGGINGFACE_TOKEN = "hf_xxx..."
        ANTHROPIC_API_KEY = "sk-ant-xxx..."  # Optional, for LLM features
        VOYAGE_API_KEY = "pa-xxx..."  # Optional, for Voyage embeddings
"""

import streamlit as st
import pandas as pd
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_secrets_to_env():
    """Load Streamlit secrets into environment variables."""
    secrets_loaded = []

    # Map of secret names to environment variable names
    secret_mappings = {
        "HUGGINGFACE_TOKEN": ["HUGGINGFACE_TOKEN", "HF_TOKEN"],
        "HF_TOKEN": ["HUGGINGFACE_TOKEN", "HF_TOKEN"],
        "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"],
        "VOYAGE_API_KEY": ["VOYAGE_API_KEY"],
    }

    try:
        for secret_name, env_names in secret_mappings.items():
            if secret_name in st.secrets:
                value = st.secrets[secret_name]
                for env_name in env_names:
                    os.environ[env_name] = value
                secrets_loaded.append(secret_name)
                logger.info(f"Loaded secret: {secret_name}")
    except Exception as e:
        logger.warning(f"Could not load secrets: {e}")

    return secrets_loaded


# Load secrets before importing pipeline components
secrets_loaded = load_secrets_to_env()

# Page configuration
st.set_page_config(
    page_title="Concordia Pipeline v3",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CRDSA Brand Constants
# ---------------------------------------------------------------------------
BRAND_BLUE = "#3B63A8"
BRAND_PURPLE = "#8B4DAA"
BRAND_TEAL = "#207870"
BRAND_NAVY = "#1A1F3C"
BRAND_TEXT = "#3A3A3C"
BRAND_LIGHT_BG = "#F7F8FA"
BRAND_BORDER = "#E2E2E2"
BRAND_RED = "#C0392B"
BRAND_AMBER = "#D4A017"


def inject_brand_css():
    """Inject CRDSA brand CSS overrides for a light-themed, professional UI.

    This embeds the full theme directly — no .streamlit/config.toml needed.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@600;700&display=swap');

    /* ---- Force light theme via Streamlit CSS variables ---- */
    :root, [data-testid="stAppViewContainer"],
    .stApp, [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        --primary-color: #3B63A8;
        --background-color: #FFFFFF;
        --secondary-background-color: #F7F8FA;
        --text-color: #1A1F3C;
    }
    .stApp {
        background-color: #FFFFFF !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stBottomBlockContainer"],
    [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
    }

    /* ---- Global font override ---- */
    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
        color: #3A3A3C;
        background-color: #FFFFFF;
    }

    /* ---- Headings ---- */
    h1, h2, h3 {
        font-family: 'Barlow Semi Condensed', sans-serif !important;
        color: #1A1F3C !important;
        font-weight: 600;
    }

    /* ---- Sidebar — dark navy ---- */
    section[data-testid="stSidebar"] {
        background-color: #1A1F3C;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #F0F0F1 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(240,240,241,0.15);
    }
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #F0F0F1 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        color: #F0F0F1 !important;
    }

    /* ---- Primary button — brand blue ---- */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #3B63A8 !important;
        border: none !important;
        color: white !important;
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 6px;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #2E4F8A !important;
    }

    /* ---- Secondary / download buttons — ghost style ---- */
    .stDownloadButton > button,
    .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
        background-color: transparent !important;
        border: 1.5px solid #3B63A8 !important;
        color: #3B63A8 !important;
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        border-radius: 6px;
    }
    .stDownloadButton > button:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: #EDF2FA !important;
    }

    /* ---- Tabs — brand blue underline ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #E2E2E2;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 500;
        color: #3A3A3C;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #3B63A8 !important;
        border-bottom: 3px solid #3B63A8;
        font-weight: 600;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background-color: #F7F8FA;
        border: 1px solid #E2E2E2;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* ---- Progress bar — brand gradient ---- */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #3B63A8 0%, #8B4DAA 100%) !important;
    }

    /* ---- Expander headers ---- */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        color: #1A1F3C;
    }

    /* ---- Alert accent colors ---- */
    .stAlert [data-testid="stNotificationContentSuccess"] { border-left-color: #207870 !important; }
    .stAlert [data-testid="stNotificationContentError"]   { border-left-color: #C0392B !important; }
    .stAlert [data-testid="stNotificationContentWarning"] { border-left-color: #D4A017 !important; }
    .stAlert [data-testid="stNotificationContentInfo"]    { border-left-color: #3B63A8 !important; }

    /* ---- Dividers ---- */
    hr { border-color: #E2E2E2 !important; }

    /* ---- File uploader area ---- */
    [data-testid="stFileUploader"] {
        border-radius: 8px;
    }

    /* ---- Dataframe container ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E2E2;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Brand UI Components
# ---------------------------------------------------------------------------

def status_badge(label: str, level: str = "info") -> str:
    """Return an HTML badge matching CRDSA brand colors."""
    colors = {
        "success": (BRAND_TEAL, "#E8F5F3"),
        "error":   (BRAND_RED, "#FDEDEB"),
        "warning": (BRAND_AMBER, "#FFF8E1"),
        "info":    (BRAND_BLUE, "#EDF2FA"),
        "purple":  (BRAND_PURPLE, "#F5EDF8"),
        "neutral": ("#6B7280", "#F3F4F6"),
    }
    fg, bg = colors.get(level, colors["info"])
    return (
        f'<span style="background:{bg}; color:{fg}; padding:4px 12px; '
        f'border-radius:4px; font-family:Barlow Semi Condensed,sans-serif; '
        f'font-weight:600; font-size:0.85rem; display:inline-block;">{label}</span>'
    )


def metric_card(label: str, value: str, accent: str = BRAND_BLUE) -> str:
    """Render a metric as a branded card with colored top-border accent."""
    return f"""
    <div style="
        background: white;
        border: 1px solid {BRAND_BORDER};
        border-top: 3px solid {accent};
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <p style="margin:0; font-size:0.75rem; color:#6B7280; font-family:Barlow,sans-serif;
                   text-transform:uppercase; letter-spacing:0.06em;">{label}</p>
        <p style="margin:6px 0 0; font-size:1.35rem; font-weight:700;
                   font-family:Barlow Semi Condensed,sans-serif; color:{BRAND_NAVY};">{value}</p>
    </div>
    """


def stoplight_panel(level: str, reason: str = "",
                    core_present: list = None, core_missing: list = None) -> str:
    """Render LLM review stoplight as a professional left-border panel."""
    color_map = {"GREEN": BRAND_TEAL, "YELLOW": BRAND_AMBER, "RED": BRAND_RED}
    bg_map = {"GREEN": "#E8F5F3", "YELLOW": "#FFF8E1", "RED": "#FDEDEB"}
    level_upper = level.upper()
    color = color_map.get(level_upper, BRAND_BLUE)
    bg = bg_map.get(level_upper, "#EDF2FA")

    html = f"""
    <div style="background:{bg}; border-left:4px solid {color};
                border-radius:0 8px 8px 0; padding:20px; margin:12px 0;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="width:14px; height:14px; border-radius:50%;
                         background:{color}; display:inline-block;"></span>
            <span style="font-family:Barlow Semi Condensed,sans-serif; font-weight:700;
                         font-size:1.1rem; color:{color};">
                REVIEW: {level_upper}
            </span>
        </div>
    """
    if reason:
        html += f'<p style="margin:4px 0 0; color:{BRAND_TEXT}; font-family:Barlow,sans-serif;">{reason}</p>'

    if core_present:
        html += (
            f'<p style="margin:10px 0 2px; font-size:0.8rem; color:#6B7280; '
            f'text-transform:uppercase; letter-spacing:0.05em;">Core Variables Present</p>'
            f'<p style="margin:0; color:{BRAND_NAVY}; font-weight:500;">{", ".join(core_present)}</p>'
        )
    if core_missing:
        html += (
            f'<p style="margin:10px 0 2px; font-size:0.8rem; color:#6B7280; '
            f'text-transform:uppercase; letter-spacing:0.05em;">Core Variables Missing</p>'
            f'<p style="margin:0; color:{BRAND_RED}; font-weight:500;">{", ".join(core_missing)}</p>'
        )
    html += "</div>"
    return html


def pipeline_stepper(stages: list, current_index: int, failed: bool = False) -> str:
    """Render a horizontal pipeline stage stepper."""
    html = '<div style="display:flex; gap:6px; margin:16px 0;">'
    for i, stage in enumerate(stages):
        if failed and i == current_index:
            color, bg = BRAND_RED, "#FDEDEB"
        elif i < current_index:
            color, bg = BRAND_TEAL, "#E8F5F3"
        elif i == current_index:
            color, bg = BRAND_BLUE, "#EDF2FA"
        else:
            color, bg = "#9CA3AF", "#F3F4F6"
        html += f'''
        <div style="flex:1; text-align:center; padding:10px 4px;
                    background:{bg}; border-radius:6px; border:1px solid {color}33;">
            <span style="font-family:Barlow Semi Condensed,sans-serif; font-weight:600;
                         font-size:0.75rem; color:{color}; letter-spacing:0.03em;">{stage}</span>
        </div>'''
    html += '</div>'
    return html

# Import pipeline components
try:
    from orchestrator import PipelineOrchestrator, create_orchestrator, PipelineResult
    from config.settings import get_settings, reset_settings
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)


def parse_data_dictionary(df: pd.DataFrame) -> Dict:
    """
    Parse a data dictionary DataFrame using deterministic rules (v2 approach).

    This uses the proven parsing logic from v2 that correctly handles
    clinical data dictionaries with continuation rows for code values.

    Output format:
    {
        "SEX": {"codes": {"1": "Male", "2": "Female"}},
        "RACE": {"codes": {"11": "White", "12": "Black or African American"}}
    }
    """
    dictionary = {}

    # Find the header row by looking for known column headers
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

    # If we found a header row, reindex the DataFrame
    if header_row_idx is not None:
        new_columns = df.iloc[header_row_idx].tolist()
        df = df.iloc[header_row_idx + 1:].copy()
        df.columns = [str(c).strip() if pd.notna(c) else f'col_{i}' for i, c in enumerate(new_columns)]

    # Identify key columns by mapping column names (case-insensitive)
    col_map = {str(c).upper().replace('\n', ' '): c for c in df.columns}

    var_col = None
    value_col = None
    format_col = None

    # Find variable name column
    for candidate in ['VARIABLE NAME', 'VARIABLE', 'VAR', 'NAME', 'FIELD']:
        if candidate in col_map:
            var_col = col_map[candidate]
            break

    # Find value/decode column
    for candidate in ['VALID VALUES', 'VALUES', 'DECODE', 'VALID VALUE']:
        if candidate in col_map:
            value_col = col_map[candidate]
            break

    # Find format column
    for candidate in ['FORMAT  (VALUE LIST)', 'FORMAT (VALUE LIST)', 'FORMAT', 'VALUE LIST', 'CODELIST']:
        if candidate in col_map:
            format_col = col_map[candidate]
            break

    if not var_col:
        logger.warning("Could not find variable name column in dictionary")
        return {}

    # Parse the dictionary - track current variable for continuation rows
    current_var = None

    for _, row in df.iterrows():
        var_name = row.get(var_col)

        # If we have a variable name, start tracking a new variable
        if pd.notna(var_name) and str(var_name).strip():
            current_var = str(var_name).strip().upper()
            if current_var not in dictionary:
                dictionary[current_var] = {
                    "codes": {},
                    "format": str(row.get(format_col, '')) if format_col and pd.notna(row.get(format_col)) else ''
                }

        # Parse code = value patterns from the value column
        if current_var and value_col:
            value_str = row.get(value_col)
            if pd.notna(value_str):
                value_str = str(value_str).strip()
                if '=' in value_str:
                    parts = value_str.split('=', 1)
                    code = parts[0].strip()
                    label = parts[1].strip()
                    dictionary[current_var]["codes"][code] = label

    # Remove variables with no codes
    dictionary = {k: v for k, v in dictionary.items() if v.get("codes")}

    logger.info(f"Deterministic parser found {len(dictionary)} variables with codes")
    for var, data in dictionary.items():
        codes = data.get("codes", {})
        logger.info(f"  {var}: {len(codes)} codes - keys: {list(codes.keys())}")

    return dictionary


def init_session_state():
    """Initialize session state variables."""
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None


def progress_callback(stage: str, status: str, message: str, progress: float):
    """Legacy callback for pipeline progress updates (kept for compatibility)."""
    st.session_state.current_stage = stage
    st.session_state.progress_log.append({
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "status": status,
        "message": message,
        "progress": progress
    })


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        # Branded sidebar header
        st.markdown("""
        <div style="text-align:center; padding:4px 0 20px;">
            <p style="font-family:Barlow Semi Condensed,sans-serif; font-weight:700;
                      font-size:1.25rem; color:#F0F0F1; margin:0; letter-spacing:0.04em;">CONCORDIA</p>
            <p style="font-size:0.75rem; color:#8B9DC3; margin:2px 0 0;
                      font-family:Barlow,sans-serif;">Pipeline v3</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("RAG Settings")
        use_rag = st.checkbox(
            "Enable RAG",
            value=True,
            help="Use RAG for dynamic specification retrieval"
        )

        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["local", "mock"],
            index=0,
            help="Local uses sentence-transformers, mock for testing"
        )

        st.subheader("LLM Settings")
        use_llm = st.checkbox(
            "Enable LLM",
            value=True,
            help="Use Claude for value resolution and review"
        )

        enable_review = st.checkbox(
            "Enable LLM Review",
            value=True,
            disabled=not use_llm,
            help="Run LLM-powered quality review stage"
        )

        st.subheader("Pipeline Options")
        skip_qc = st.checkbox(
            "Skip QC Stage",
            value=False,
            help="Skip quality control checks"
        )

        st.divider()

        st.markdown("""
        <div style="font-family:Barlow,sans-serif; font-size:0.85rem; color:#C0C8D8; line-height:1.6;">
            <p style="margin:0 0 6px; font-weight:600; color:#F0F0F1;">About</p>
            RAG-enhanced harmonization for SDTM Demographics (DM) domain.
            Dynamic spec retrieval, LLM-powered resolution, 18 output variables,
            automated QC, review stage, and full lineage tracking.
        </div>
        """, unsafe_allow_html=True)

        # API key status — clean indicators
        st.divider()
        st.subheader("API Keys")

        hf_configured = bool(os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"))
        anthropic_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
        voyage_configured = bool(os.environ.get("VOYAGE_API_KEY"))

        def _key_dot(configured, optional=False):
            if configured:
                return f'<span style="color:{BRAND_TEAL}; font-weight:600;">Connected</span>'
            elif optional:
                return '<span style="color:#6B7280;">Optional</span>'
            else:
                return f'<span style="color:{BRAND_AMBER}; font-weight:600;">Missing</span>'

        st.markdown(
            f'<div style="font-family:Barlow,sans-serif; font-size:0.85rem; line-height:2; color:#C0C8D8;">'
            f'HuggingFace: {_key_dot(hf_configured)}<br>'
            f'Anthropic: {_key_dot(anthropic_configured, optional=True)}<br>'
            f'Voyage AI: {_key_dot(voyage_configured, optional=True)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not hf_configured:
            st.caption("Add HUGGINGFACE_TOKEN to .streamlit/secrets.toml")

        return {
            "use_rag": use_rag,
            "use_llm": use_llm,
            "enable_review": enable_review,
            "embedding_provider": embedding_provider,
            "skip_qc": skip_qc
        }


def render_file_upload():
    """Render file upload section."""
    st.subheader("Input Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload source data file",
            type=["csv", "xlsx", "xls", "sas7bdat"],
            help="Supported formats: CSV, Excel, SAS"
        )

    with col2:
        trial_id = st.text_input(
            "Trial ID (optional)",
            placeholder="e.g., NCT12345678",
            help="Leave blank to extract from filename"
        )

    # Optional data dictionary upload
    with st.expander("Data Dictionary (Optional)", expanded=False):
        st.caption("Upload a data dictionary to improve column mapping accuracy")
        data_dict_file = st.file_uploader(
            "Upload data dictionary",
            type=["csv", "xlsx", "xls", "json"],
            help="Maps source column names to descriptions or standard terms",
            key="data_dict_uploader"
        )

        if data_dict_file:
            st.success(f"Data dictionary loaded: {data_dict_file.name}")

    return uploaded_file, trial_id, data_dict_file if 'data_dict_file' in dir() else None


def render_progress():
    """Render progress indicators with horizontal pipeline stepper."""
    if st.session_state.progress_log:
        latest = st.session_state.progress_log[-1]

        # Determine current stage index for the stepper
        stage_order = ["INGEST", "MAP", "HARMONIZE", "QC", "REVIEW"]
        current_stage = latest["stage"].upper()
        is_error = current_stage == "ERROR"
        is_complete = current_stage in ("COMPLETE", "FINALIZE")

        if is_complete:
            current_idx = len(stage_order)  # all stages done
        elif current_stage in stage_order:
            current_idx = stage_order.index(current_stage)
        else:
            current_idx = 0

        # Render the horizontal stepper
        st.markdown(
            pipeline_stepper(stage_order, current_idx, failed=is_error),
            unsafe_allow_html=True
        )

        # Progress bar
        st.progress(latest["progress"])

        # Status message — use badge instead of emoji
        level = "error" if is_error else ("success" if is_complete else "info")
        badge = status_badge(latest["stage"].upper(), level)
        st.markdown(
            f'{badge} <span style="margin-left:10px; font-family:Barlow,sans-serif; '
            f'color:{BRAND_TEXT};">{latest["message"]}</span>',
            unsafe_allow_html=True
        )


def create_results_zip(result: PipelineResult) -> bytes:
    """Create a ZIP file containing all pipeline results.

    Includes:
    - Harmonized data CSV
    - QC report CSV
    - Transformation report DOCX (with row-by-row details)

    Excludes JSON files per user request.
    """
    import io
    import zipfile

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        trial_id = result.metadata.get("trial_id", "output")

        # Add harmonized data
        if result.harmonized_data is not None:
            csv_data = result.harmonized_data.to_csv(index=False)
            zf.writestr(f"{trial_id}_harmonized.csv", csv_data)

        # Add QC report
        if result.qc_report is not None and len(result.qc_report) > 0:
            qc_csv = result.qc_report.to_csv(index=False)
            zf.writestr(f"{trial_id}_qc_report.csv", qc_csv)

        # Add transformation report as DOCX (v2 format with row-by-row summary)
        try:
            docx_report = create_transformation_report_docx(result)
            zf.writestr(f"{trial_id}_transformation_report.docx", docx_report)
        except ImportError:
            logger.warning("python-docx not installed, skipping DOCX report")
        except Exception as e:
            logger.warning(f"Could not create DOCX report: {e}")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def create_transformation_report_docx(result: PipelineResult) -> bytes:
    """
    Create a DOCX transformation report matching v2 format exactly.

    Returns bytes for the DOCX file.
    """
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    import io

    doc = Document()

    # Title (v2 format)
    title = doc.add_heading('Harmonization Transformation Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Get metadata
    ingest_meta = result.metadata.get('stages', {}).get('ingest', {})
    harmonize_meta = result.metadata.get('stages', {}).get('harmonize', {})

    trial_id = result.metadata.get('trial_id', 'Unknown')
    source_filename = ingest_meta.get('source_filename', result.metadata.get('source_filename', 'Unknown'))
    rows_in = ingest_meta.get('rows', result.metadata.get('rows_processed', 0))
    rows_out = harmonize_meta.get('rows_out', rows_in)
    rows_dropped = rows_in - rows_out if rows_in and rows_out else 0

    # Header info (v2 format - not a table, just paragraphs)
    doc.add_paragraph(f"Trial: {trial_id}")
    doc.add_paragraph(f"Input: {source_filename}")
    doc.add_paragraph(f"Rows in input: {rows_in} | Rows in output: {rows_out} | Rows dropped: {rows_dropped}")
    doc.add_paragraph(f"Pipeline version: v3 (RAG-Enhanced Agentic Architecture)")
    doc.add_paragraph(f"Run ID: {result.metadata.get('timestamp', 'Unknown')}")

    # Execution Summary (v2 format)
    doc.add_heading('Execution Summary', level=1)
    stages = result.metadata.get('stages', {})
    for stage_name, stage_info in stages.items():
        if isinstance(stage_info, dict):
            status = "✓" if stage_info.get('success', True) else "✗"
            time_ms = stage_info.get('execution_time_ms', 0)
            doc.add_paragraph(f"{status} {stage_name}: {time_ms}ms")

    # Dictionary info
    dict_filename = ingest_meta.get('dictionary_filename') or result.metadata.get('dictionary_filename')
    if dict_filename:
        doc.add_paragraph(f"Dictionary used: {dict_filename}")
    else:
        doc.add_paragraph("Dictionary used: None")

    doc.add_paragraph()

    # Output Schema (v2 format)
    doc.add_heading('1. Output Schema', level=1)
    if result.harmonized_data is not None:
        doc.add_paragraph(', '.join(result.harmonized_data.columns.tolist()))
    else:
        doc.add_paragraph("TRIAL, SUBJID, SEX, RACE, AGE, AGEU, AGEGP, ETHNIC, COUNTRY, SITEID, STUDYID, USUBJID, ARMCD, ARM, BRTHDTC, RFSTDTC, RFENDTC, DOMAIN")

    # Variable Transformation Table (v2 format)
    doc.add_heading('2. Variable-Level Transformations', level=1)

    if result.lineage:
        trans_table = doc.add_table(rows=1, cols=7)
        trans_table.style = 'Table Grid'

        # Headers (v2 format)
        headers = ['Variable', 'Source', 'Operation', 'Details', 'Changed', '%', 'Missing']
        hdr_cells = trans_table.rows[0].cells
        for i, header in enumerate(headers):
            hdr_cells[i].text = header

        # Data rows
        for entry in result.lineage:
            row_cells = trans_table.add_row().cells
            row_cells[0].text = str(entry.get('variable', ''))

            # Source column - show (derived) if None
            source_col = entry.get('source_column', '') or ''
            row_cells[1].text = str(source_col) if source_col else '(derived)'

            row_cells[2].text = str(entry.get('mapping_operation', 'Copy'))

            # Use transform_operation (v2 key name), truncated to 50 chars
            transform_op = entry.get('transform_operation', '') or entry.get('transformation', 'None')
            row_cells[3].text = str(transform_op)[:50]

            row_cells[4].text = str(entry.get('rows_changed', 0))
            row_cells[5].text = f"{entry.get('percent_changed', 0):.1f}%"
            row_cells[6].text = str(entry.get('missing_count', 0))

    # QC Report Section (v2 format)
    doc.add_heading('3. QC Report', level=1)

    if result.qc_report is not None and len(result.qc_report) > 0:
        qc_table = doc.add_table(rows=1, cols=5)
        qc_table.style = 'Table Grid'

        qc_headers = ['TRIAL', 'Issue Type', 'Variable', 'Rows Affected', 'Notes']
        hdr_cells = qc_table.rows[0].cells
        for i, header in enumerate(qc_headers):
            hdr_cells[i].text = header

        for _, issue in result.qc_report.iterrows():
            row_cells = qc_table.add_row().cells
            row_cells[0].text = str(trial_id)
            row_cells[1].text = str(issue.get('issue_type', ''))
            row_cells[2].text = str(issue.get('variable', ''))
            row_cells[3].text = str(issue.get('n_rows_affected', 0))
            row_cells[4].text = str(issue.get('notes', ''))[:100]
    else:
        doc.add_paragraph("No QC issues found.")

    # Files Produced (v2 format)
    doc.add_heading('4. Files Produced', level=1)
    doc.add_paragraph(f"Harmonized output: {trial_id}_DM_harmonized_*.csv")
    doc.add_paragraph(f"QC report: {trial_id}_QC_report_*.csv")
    doc.add_paragraph(f"Transformation report: This document")

    # LLM Review section (v3 addition)
    if result.review_result:
        doc.add_heading('5. LLM Review', level=1)

        review = result.review_result
        doc.add_paragraph(f"Approval: {review.get('approval', 'Unknown')}")
        doc.add_paragraph(f"Quality: {review.get('overall_quality', 'Unknown')}")

        if review.get('reason'):
            doc.add_paragraph(f"Summary: {review.get('reason')}")

    # Warnings and Errors
    if result.warnings:
        doc.add_heading('Warnings', level=1)
        for w in result.warnings:
            doc.add_paragraph(f"• {w}", style='List Bullet')

    if result.errors:
        doc.add_heading('Errors', level=1)
        for e in result.errors:
            doc.add_paragraph(f"• {e}", style='List Bullet')

    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def render_results(result: PipelineResult):
    """Render pipeline results with branded card metrics and stoplight panel."""
    st.header("Results")

    # --- Row 1: Summary metric cards (colored top-border accents) ---
    rows = len(result.harmonized_data) if result.harmonized_data is not None else 0
    qc_issues = len(result.qc_report) if result.qc_report is not None else 0
    time_sec = result.execution_time_ms / 1000

    status_text = "Success" if result.success else "Failed"
    status_color = BRAND_TEAL if result.success else BRAND_RED

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Status", status_text, status_color), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Rows Processed", str(rows), BRAND_BLUE), unsafe_allow_html=True)
    with col3:
        qc_color = BRAND_TEAL if qc_issues == 0 else BRAND_AMBER
        st.markdown(metric_card("QC Issues", str(qc_issues), qc_color), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Execution Time", f"{time_sec:.1f}s", BRAND_PURPLE), unsafe_allow_html=True)

    # --- Row 2: LLM metric cards ---
    llm_enabled = result.metadata.get("llm_enabled", False)
    tokens = result.metadata.get("llm_tokens_used", 0)
    model = result.metadata.get("llm_model", "none")
    review_status = "N/A"
    if result.review_result:
        approval = result.review_result.get("approval", "unknown")
        review_status = approval.replace("_", " ").title()

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        llm_text = "Enabled" if llm_enabled else "Disabled"
        llm_color = BRAND_BLUE if llm_enabled else "#9CA3AF"
        st.markdown(metric_card("LLM", llm_text, llm_color), unsafe_allow_html=True)
    with col6:
        st.markdown(metric_card("LLM Tokens", f"{tokens:,}" if tokens else "-", BRAND_BLUE), unsafe_allow_html=True)
    with col7:
        rev_color = BRAND_TEAL if "green" in review_status.lower() else (BRAND_AMBER if "yellow" in review_status.lower() else BRAND_PURPLE)
        st.markdown(metric_card("Review", review_status, rev_color), unsafe_allow_html=True)
    with col8:
        st.markdown(metric_card("Model", model if model != "none" else "-", BRAND_PURPLE), unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # --- Errors and warnings ---
    if result.errors:
        st.error("**Errors:**\n" + "\n".join(f"- {e}" for e in result.errors))

    if result.warnings:
        st.warning("**Warnings:**\n" + "\n".join(f"- {w}" for w in result.warnings))

    # --- Download All as ZIP ---
    if result.success:
        zip_data = create_results_zip(result)
        if zip_data:
            trial_id = result.metadata.get("trial_id", "output")
            st.download_button(
                "Download All Results (ZIP)",
                zip_data,
                file_name=f"{trial_id}_harmonization_results.zip",
                mime="application/zip",
                type="primary"
            )

    # --- Tabs for detailed results ---
    tabs = st.tabs(["Harmonized Data", "QC Report", "LLM Review", "Transformations", "Downloads"])

    with tabs[0]:
        if result.harmonized_data is not None:
            st.dataframe(result.harmonized_data, use_container_width=True)
            csv = result.harmonized_data.to_csv(index=False)
            st.download_button(
                "Download Harmonized Data",
                csv,
                file_name="harmonized_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No harmonized data available")

    with tabs[1]:
        if result.qc_report is not None and len(result.qc_report) > 0:
            st.dataframe(result.qc_report, use_container_width=True)
            csv = result.qc_report.to_csv(index=False)
            st.download_button(
                "Download QC Report",
                csv,
                file_name="qc_report.csv",
                mime="text/csv"
            )
        else:
            st.success("No QC issues found.")

    with tabs[2]:
        if result.review_result:
            review = result.review_result

            # Branded stoplight panel (replaces emoji-based display)
            stoplight_level = review.get("stoplight") or review.get("approval", "unknown")
            reason = review.get("reason", "")
            core_present = review.get("core_variables_present", [])
            core_missing = review.get("core_variables_missing", [])

            st.markdown(
                stoplight_panel(stoplight_level, reason, core_present, core_missing or None),
                unsafe_allow_html=True,
            )

            # Quality assessment
            if "overall_quality" in review:
                st.markdown(
                    f'<p style="margin:12px 0 4px; font-size:0.8rem; color:#6B7280; '
                    f'text-transform:uppercase; letter-spacing:0.05em;">Quality Rating</p>'
                    f'<p style="color:{BRAND_NAVY}; font-weight:600; font-family:Barlow Semi Condensed,sans-serif;">'
                    f'{review["overall_quality"]}</p>',
                    unsafe_allow_html=True,
                )

            # Formatting issues
            if review.get("formatting_issues"):
                with st.expander("Formatting Issues"):
                    for issue in review["formatting_issues"]:
                        st.markdown(f"- {issue}")

            # Critical issues
            if review.get("critical_issues"):
                with st.expander("Critical Issues"):
                    for issue in review["critical_issues"]:
                        if isinstance(issue, dict):
                            st.markdown(f"- {issue.get('issue', issue)}")
                        else:
                            st.markdown(f"- {issue}")

            # Recommendations
            if review.get("recommendations"):
                with st.expander("Recommendations"):
                    for rec in review["recommendations"]:
                        st.markdown(f"- {rec}")

            # Full details
            with st.expander("Full Review Details"):
                st.json(review)
        else:
            st.info("No LLM review performed (requires Anthropic API key)")

    with tabs[3]:
        st.subheader("Variable Transformations")
        if result.lineage:
            lineage_data = []
            for entry in result.lineage:
                lineage_data.append({
                    "variable": entry.get("variable", ""),
                    "source_column": entry.get("source_column", "") or "(derived)",
                    "mapping_operation": entry.get("mapping_operation", ""),
                    "transform_operation": entry.get("transform_operation", "") or entry.get("transformation", ""),
                    "transform_details": str(entry.get("transform_details", {})),
                    "rows_changed": entry.get("rows_changed", 0),
                    "percent_changed": f"{entry.get('percent_changed', 0):.1f}%",
                    "missing_count": entry.get("missing_count", 0),
                    "non_null_count": entry.get("non_null_count", 0)
                })
            lineage_df = pd.DataFrame(lineage_data)
            st.dataframe(lineage_df, use_container_width=True)
        else:
            st.info("No transformation data available")

    with tabs[4]:
        st.subheader("Downloads")
        col1, col2 = st.columns(2)

        with col1:
            if result.harmonized_data is not None:
                trial_id = result.metadata.get("trial_id", "output")
                csv = result.harmonized_data.to_csv(index=False)
                st.download_button(
                    "Harmonized Data (CSV)",
                    csv,
                    file_name=f"{trial_id}_harmonized.csv",
                    mime="text/csv"
                )

            if result.qc_report is not None and len(result.qc_report) > 0:
                qc_csv = result.qc_report.to_csv(index=False)
                st.download_button(
                    "QC Report (CSV)",
                    qc_csv,
                    file_name=f"{trial_id}_qc_report.csv",
                    mime="text/csv"
                )

        with col2:
            try:
                docx_bytes = create_transformation_report_docx(result)
                st.download_button(
                    "Transformation Report (DOCX)",
                    docx_bytes,
                    file_name=f"{trial_id}_transformation_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.warning(f"Could not generate DOCX report: {e}")

            if result.success:
                zip_data = create_results_zip(result)
                if zip_data:
                    st.download_button(
                        "All Results (ZIP)",
                        zip_data,
                        file_name=f"{trial_id}_harmonization_results.zip",
                        mime="application/zip"
                    )

    # Metadata
    with st.expander("Metadata"):
        st.json(result.metadata)


def run_pipeline(uploaded_file, trial_id: str, config: dict, data_dict_file=None):
    """Run the harmonization pipeline."""
    # Reset progress
    st.session_state.progress_log = []
    st.session_state.pipeline_result = None

    # Read data dictionary if provided - we'll parse it with LLM after creating orchestrator
    dict_df = None
    data_dict = None
    if data_dict_file is not None:
        try:
            if data_dict_file.name.endswith('.csv'):
                dict_df = pd.read_csv(data_dict_file)
            elif data_dict_file.name.endswith(('.xlsx', '.xls')):
                dict_df = pd.read_excel(data_dict_file)
            elif data_dict_file.name.endswith('.json'):
                # JSON is already structured, use directly
                data_dict = json.load(data_dict_file)
            logger.info(f"Loaded dictionary file: {data_dict_file.name}")
        except Exception as e:
            st.warning(f"Could not load data dictionary: {e}")
            logger.exception("Data dictionary load error")

    # Read uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.sas7bdat'):
            # SAS7BDAT format requires pyreadstat or sas7bdat package
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sas7bdat(uploaded_file)
                logger.info(f"Loaded SAS file with {len(df)} rows using pyreadstat")
            except ImportError:
                try:
                    from sas7bdat import SAS7BDAT
                    with SAS7BDAT(uploaded_file) as f:
                        df = f.to_data_frame()
                    logger.info(f"Loaded SAS file with {len(df)} rows using sas7bdat")
                except ImportError:
                    st.error("SAS7BDAT support requires 'pyreadstat' or 'sas7bdat' package. "
                             "Install with: pip install pyreadstat")
                    return None
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # Extract trial_id from filename if not provided
    if not trial_id:
        import re
        match = re.search(r'(NCT\d{8})', uploaded_file.name)
        trial_id = match.group(1) if match else Path(uploaded_file.name).stem

    # Create progress display elements
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_text = st.empty()

    # Create a progress callback that updates Streamlit elements
    def streamlit_progress_callback(stage: str, status: str, message: str, progress: float):
        """Update Streamlit progress elements in real-time."""
        # Update progress bar (clamp to 0-1 range)
        progress_value = max(0.0, min(1.0, progress))
        progress_bar.progress(progress_value, text=f"{stage.upper()}: {message}")

        # Also log to session state for history
        st.session_state.progress_log.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "message": message,
            "progress": progress
        })

    # Create orchestrator with real-time progress callback
    reset_settings()
    orchestrator = create_orchestrator(
        use_rag=config["use_rag"],
        use_llm=config["use_llm"],
        enable_review=config["enable_review"],
        progress_callback=streamlit_progress_callback,
        embedding_provider=config["embedding_provider"]
    )

    # Parse data dictionary using deterministic parser (v2 approach)
    if dict_df is not None and data_dict is None:
        progress_bar.progress(0.05, text="INIT: Parsing data dictionary...")

        try:
            data_dict = parse_data_dictionary(dict_df)
            if data_dict:
                st.success(f"Dictionary parsed: {len(data_dict)} variables extracted")
                logger.info(f"Parsed dictionary variables: {list(data_dict.keys())}")
                for var, info in data_dict.items():
                    logger.info(f"  {var}: {list(info.get('codes', {}).keys())}")
            else:
                st.warning("Could not extract code mappings from dictionary")
        except Exception as e:
            logger.exception("Failed to parse dictionary")
            st.warning(f"Dictionary parsing failed: {e}")

    # Run pipeline (no spinner - we have the progress bar)
    try:
        result = orchestrator.run(
            input_df=df,
            trial_id=trial_id,
            skip_qc=config["skip_qc"],
            data_dict=data_dict
        )
    except Exception as e:
        logger.exception("Pipeline run failed with exception")
        st.error(f"Pipeline error: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

    # Clear progress elements after completion
    if result.success:
        progress_bar.progress(1.0, text="Pipeline complete")
    else:
        progress_bar.progress(1.0, text="Pipeline failed")
        if result.errors:
            for err in result.errors:
                st.error(f"Error: {err}")

    st.session_state.pipeline_result = result
    return result


def main():
    """Main application entry point."""
    init_session_state()

    # Inject brand CSS
    inject_brand_css()

    # Branded header
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:4px;">
        <div>
            <h1 style="margin:0; font-size:1.8rem; font-family:Barlow Semi Condensed,sans-serif;
                        font-weight:700; color:{BRAND_NAVY} !important;">Concordia Pipeline v3</h1>
            <p style="margin:2px 0 0; color:{BRAND_BLUE}; font-weight:500;
                      font-family:Barlow,sans-serif; font-size:0.95rem;">
                RAG-Enhanced Clinical Trial Data Harmonization
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not PIPELINE_AVAILABLE:
        st.error(f"Pipeline components not available: {IMPORT_ERROR}")
        st.info("Please ensure all dependencies are installed.")
        return

    # Sidebar configuration
    config = render_sidebar()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file, trial_id, data_dict_file = render_file_upload()

        if uploaded_file is not None:
            # Show file preview
            with st.expander("File Preview", expanded=True):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        preview_df = pd.read_csv(uploaded_file, nrows=5)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        preview_df = pd.read_excel(uploaded_file, nrows=5)
                    elif uploaded_file.name.endswith('.sas7bdat'):
                        try:
                            import pyreadstat
                            preview_df, _ = pyreadstat.read_sas7bdat(uploaded_file)
                            preview_df = preview_df.head(5)
                        except ImportError:
                            try:
                                from sas7bdat import SAS7BDAT
                                with SAS7BDAT(uploaded_file) as f:
                                    preview_df = f.to_data_frame().head(5)
                            except ImportError:
                                st.warning("Install 'pyreadstat' to preview SAS files")
                                preview_df = None
                    else:
                        preview_df = None

                    if preview_df is not None:
                        st.dataframe(preview_df, use_container_width=True)
                        st.caption(f"Showing first 5 rows | {len(preview_df.columns)} columns")

                    # Reset file position for later reading
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing file: {e}")

    with col2:
        st.subheader("Run Pipeline")

        if uploaded_file is not None:
            if st.button("Start Harmonization", type="primary", use_container_width=True):
                result = run_pipeline(uploaded_file, trial_id, config, data_dict_file)
        else:
            st.info("Upload a file to begin")

    # Results section
    st.divider()

    if st.session_state.pipeline_result is not None:
        render_results(st.session_state.pipeline_result)


if __name__ == "__main__":
    main()
