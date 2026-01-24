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

# Import pipeline components
try:
    from orchestrator import PipelineOrchestrator, create_orchestrator, PipelineResult
    from config.settings import get_settings, reset_settings
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)


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
        st.title("⚙️ Configuration")

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

        st.subheader("About")
        st.markdown("""
        **Concordia Pipeline v3**

        RAG-enhanced harmonization for
        SDTM Demographics (DM) domain.

        Features:
        - 🔍 Dynamic spec retrieval
        - 🤖 LLM-powered resolution
        - 📊 18 output variables
        - ✅ Automated QC checks
        - 🔎 LLM Review stage
        - 📝 Full lineage tracking
        """)

        # Show secrets status
        st.divider()
        st.subheader("🔑 API Keys")

        hf_configured = bool(os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"))
        anthropic_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
        voyage_configured = bool(os.environ.get("VOYAGE_API_KEY"))

        st.markdown(f"- HuggingFace: {'✅' if hf_configured else '❌'}")
        st.markdown(f"- Anthropic: {'✅' if anthropic_configured else '⚪ (optional)'}")
        st.markdown(f"- Voyage AI: {'✅' if voyage_configured else '⚪ (optional)'}")

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
    st.subheader("📁 Input Data")

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
    with st.expander("📖 Data Dictionary (Optional)", expanded=False):
        st.caption("Upload a data dictionary to improve column mapping accuracy")
        data_dict_file = st.file_uploader(
            "Upload data dictionary",
            type=["csv", "xlsx", "xls", "json"],
            help="Maps source column names to descriptions or standard terms",
            key="data_dict_uploader"
        )

        if data_dict_file:
            st.success(f"✅ Data dictionary loaded: {data_dict_file.name}")

    return uploaded_file, trial_id, data_dict_file if 'data_dict_file' in dir() else None


def render_progress():
    """Render progress indicators."""
    if st.session_state.progress_log:
        latest = st.session_state.progress_log[-1]

        # Progress bar
        progress_bar = st.progress(latest["progress"])

        # Status message
        stage_emoji = {
            "init": "🔧",
            "ingest": "📥",
            "map": "🗺️",
            "harmonize": "✨",
            "qc": "✅",
            "review": "🔎",
            "finalize": "📦",
            "complete": "🎉",
            "error": "❌"
        }
        emoji = stage_emoji.get(latest["stage"], "⏳")
        st.info(f"{emoji} **{latest['stage'].upper()}**: {latest['message']}")


def render_results(result: PipelineResult):
    """Render pipeline results."""
    st.header("📊 Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Status",
            "✅ Success" if result.success else "❌ Failed"
        )

    with col2:
        rows = len(result.harmonized_data) if result.harmonized_data is not None else 0
        st.metric("Rows Processed", rows)

    with col3:
        qc_issues = len(result.qc_report) if result.qc_report is not None else 0
        st.metric("QC Issues", qc_issues)

    with col4:
        time_sec = result.execution_time_ms / 1000
        st.metric("Time", f"{time_sec:.1f}s")

    # LLM metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        llm_enabled = result.metadata.get("llm_enabled", False)
        st.metric("LLM", "✅ Enabled" if llm_enabled else "⚪ Disabled")

    with col6:
        tokens = result.metadata.get("llm_tokens_used", 0)
        st.metric("LLM Tokens", f"{tokens:,}" if tokens else "-")

    with col7:
        review_status = "N/A"
        if result.review_result:
            approval = result.review_result.get("approval", "unknown")
            review_status = approval.replace("_", " ").title()
        st.metric("Review", review_status)

    with col8:
        model = result.metadata.get("llm_model", "none")
        st.metric("Model", model if model != "none" else "-")

    # Errors and warnings
    if result.errors:
        st.error("**Errors:**\n" + "\n".join(f"- {e}" for e in result.errors))

    if result.warnings:
        st.warning("**Warnings:**\n" + "\n".join(f"- {w}" for w in result.warnings))

    # Tabs for detailed results
    tabs = st.tabs(["📋 Harmonized Data", "⚠️ QC Report", "🔎 LLM Review", "🗺️ Mapping Log", "📝 Lineage"])

    with tabs[0]:
        if result.harmonized_data is not None:
            st.dataframe(result.harmonized_data, use_container_width=True)

            # Download button
            csv = result.harmonized_data.to_csv(index=False)
            st.download_button(
                "⬇️ Download Harmonized Data",
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
                "⬇️ Download QC Report",
                csv,
                file_name="qc_report.csv",
                mime="text/csv"
            )
        else:
            st.success("No QC issues found! ✨")

    with tabs[2]:
        if result.review_result:
            review = result.review_result

            # Show approval status
            approval = review.get("approval", "unknown")
            if approval == "approved":
                st.success(f"✅ **LLM Review: APPROVED**")
            elif approval == "approved_with_warnings":
                st.warning(f"⚠️ **LLM Review: APPROVED WITH WARNINGS**")
            elif approval == "rejected":
                st.error(f"❌ **LLM Review: REJECTED**")
            else:
                st.info(f"🔍 **LLM Review: {approval.upper()}**")

            # Show quality assessment
            if "overall_quality" in review:
                st.markdown(f"**Quality Rating:** {review['overall_quality']}")

            # Show reason
            if "reason" in review:
                st.markdown(f"**Summary:** {review['reason']}")

            # Show critical issues
            if review.get("critical_issues"):
                st.subheader("Critical Issues")
                for issue in review["critical_issues"]:
                    st.markdown(f"- {issue}")

            # Show recommendations
            if review.get("recommendations"):
                st.subheader("Recommendations")
                for rec in review["recommendations"]:
                    st.markdown(f"- {rec}")

            # Full details
            with st.expander("Full Review Details"):
                st.json(review)
        else:
            st.info("No LLM review performed (requires Anthropic API key)")

    with tabs[3]:
        if result.mapping_log:
            # Convert to DataFrame for display
            mapping_df = pd.DataFrame(result.mapping_log)
            st.dataframe(mapping_df, use_container_width=True)
        else:
            st.info("No mapping log available")

    with tabs[4]:
        if result.lineage:
            st.json(result.lineage)
        else:
            st.info("No lineage data available")

    # Metadata
    with st.expander("📋 Metadata"):
        st.json(result.metadata)


def run_pipeline(uploaded_file, trial_id: str, config: dict, data_dict_file=None):
    """Run the harmonization pipeline."""
    # Reset progress
    st.session_state.progress_log = []
    st.session_state.pipeline_result = None

    # Read data dictionary if provided
    data_dict = None
    if data_dict_file is not None:
        try:
            if data_dict_file.name.endswith('.csv'):
                data_dict = pd.read_csv(data_dict_file).to_dict(orient='records')
            elif data_dict_file.name.endswith(('.xlsx', '.xls')):
                data_dict = pd.read_excel(data_dict_file).to_dict(orient='records')
            elif data_dict_file.name.endswith('.json'):
                data_dict = json.load(data_dict_file)
            logger.info(f"Loaded data dictionary with {len(data_dict)} entries")
        except Exception as e:
            st.warning(f"Could not load data dictionary: {e}")
            data_dict = None

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
        stage_emoji = {
            "init": "🔧",
            "ingest": "📥",
            "map": "🗺️",
            "harmonize": "✨",
            "qc": "✅",
            "review": "🔎",
            "finalize": "📦",
            "complete": "🎉",
            "error": "❌"
        }
        emoji = stage_emoji.get(stage, "⏳")

        # Update progress bar (clamp to 0-1 range)
        progress_value = max(0.0, min(1.0, progress))
        progress_bar.progress(progress_value, text=f"{emoji} {stage.upper()}: {message}")

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

    # Run pipeline (no spinner - we have the progress bar)
    result = orchestrator.run(
        input_df=df,
        trial_id=trial_id,
        skip_qc=config["skip_qc"],
        data_dict=data_dict
    )

    # Clear progress elements after completion
    if result.success:
        progress_bar.progress(1.0, text="🎉 Pipeline complete!")
    else:
        progress_bar.progress(1.0, text="❌ Pipeline failed")

    st.session_state.pipeline_result = result
    return result


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.title("🔬 Concordia Pipeline v3")
    st.markdown("*RAG-Enhanced Clinical Trial Data Harmonization*")

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
            with st.expander("📄 File Preview", expanded=True):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        preview_df = pd.read_csv(uploaded_file, nrows=5)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        preview_df = pd.read_excel(uploaded_file, nrows=5)
                    elif uploaded_file.name.endswith('.sas7bdat'):
                        # For SAS files, read full file then take head (pyreadstat doesn't support nrows)
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
                        st.caption(f"Showing first 5 rows • {len(preview_df.columns)} columns")

                    # Reset file position for later reading
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing file: {e}")

    with col2:
        st.subheader("🚀 Run Pipeline")

        if uploaded_file is not None:
            if st.button("▶️ Start Harmonization", type="primary", use_container_width=True):
                result = run_pipeline(uploaded_file, trial_id, config, data_dict_file)
        else:
            st.info("Upload a file to begin")

    # Results section
    st.divider()

    if st.session_state.pipeline_result is not None:
        render_results(st.session_state.pipeline_result)


if __name__ == "__main__":
    main()
