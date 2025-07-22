import streamlit as st

from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_logs_panel(training_service: TrainingService) -> None:
    """Display the live training logs panel as a code area with error count."""
    if st.session_state.abort_training:
        _display_frozen_logs()
        return
    
    _display_live_logs(training_service)


def _display_frozen_logs() -> None:
    """Display frozen logs when training is aborted."""
    log_content = st.session_state.frozen_log
    st.subheader("Live Training Logs (Frozen)")
    st.code(log_content, language="log", height=400)


def _display_live_logs(training_service: TrainingService) -> None:
    """Display live training logs with error count."""
    st.subheader("Live Training Logs")
    stdout, stderr = training_service.get_log_contents()

    log_content = stdout or "Waiting for training process to start..."
    error_count = _count_errors(stderr)

    if stderr:
        log_content += f"\n\n--- ERRORS ---\n{stderr}"

    st.metric("Error Count", error_count)
    st.code(log_content, language="log", height=400)

    # Store for frozen state
    st.session_state.frozen_log = log_content


def _count_errors(stderr: str) -> int:
    """Count errors in stderr using multiple patterns."""
    if not stderr:
        return 0

    stderr_lower = stderr.lower()
    return stderr_lower.count("error")
