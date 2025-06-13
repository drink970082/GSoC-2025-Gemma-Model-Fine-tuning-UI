import time

import pandas as pd
import streamlit as st

from config.training_config import DEFAULT_MODEL_CONFIG

@st.fragment(run_every=1)
def display_kpis(event_data: dict[str, pd.DataFrame]):
    """Display the key performance indicators panel."""
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(4)
    model_config = DEFAULT_MODEL_CONFIG
    total_steps = int(model_config.get("epochs", 0))
    latest_step, latest_loss, throughput, eta_str = 0, 0.0, 0.0, "N/A"

    if "loss" in event_data and not event_data["loss"].empty:
        loss_df = event_data["loss"]
        latest_event = loss_df.iloc[-1]
        latest_step, latest_loss = latest_event["step"], latest_event["value"]

        if len(loss_df) > 1:
            last_step_time = (
                loss_df.iloc[-1]["wall_time"] - loss_df.iloc[-2]["wall_time"]
            )
            if last_step_time > 0:
                throughput = 1 / last_step_time
            avg_step_time = (
                loss_df.iloc[-1]["wall_time"] - loss_df.iloc[0]["wall_time"]
            ) / len(loss_df)
            remaining_steps = total_steps - latest_step
            if remaining_steps > 0 and avg_step_time > 0:
                eta_seconds = remaining_steps * avg_step_time
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

    kpi_cols[0].metric("Global Step", f"{latest_step}/{total_steps}")
    kpi_cols[1].metric("Current Loss", f"{latest_loss:.4f}")
    kpi_cols[2].metric("Throughput", f"{throughput:.2f} steps/sec")
    kpi_cols[3].metric("ETA", eta_str)
