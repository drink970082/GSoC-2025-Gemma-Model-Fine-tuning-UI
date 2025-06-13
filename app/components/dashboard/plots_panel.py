import pandas as pd
import streamlit as st

@st.fragment(run_every=1)
def display_native_plots(event_data: dict[str, pd.DataFrame]):
    """Display the core performance plots panel."""
    st.subheader("Core Performance Plots")
    if not event_data:
        st.info("Waiting for first metric to be logged...")
        return

    for metric_name, metric_df in event_data.items():
        st.markdown(f"**Metric:** `{metric_name}`")
        st.line_chart(metric_df, x="step", y="value", use_container_width=True)
