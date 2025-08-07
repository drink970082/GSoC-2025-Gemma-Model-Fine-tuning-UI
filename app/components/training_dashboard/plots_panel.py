import pandas as pd
import streamlit as st

from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_plots_panel(training_service: TrainingService) -> None:
    """Display the core performance plots panel."""
    if st.session_state.abort_training:
        training_metrics = st.session_state.frozen_training_metrics
    else:
        training_metrics = training_service.get_training_metrics()
        st.session_state.frozen_training_metrics = training_metrics
    if not training_metrics:
        st.info("Waiting for first metric to be logged...")
        return

    loss_metrics: dict[str, pd.DataFrame] = {}
    perf_metrics: dict[str, pd.DataFrame] = {}
    # Create loss plots
    for key, value in training_metrics.items():
        if key.startswith("losses/"):
            loss_metrics[key] = value
        elif key.startswith("perf_stats/"):
            perf_metrics[key] = value

    if loss_metrics:
        _create_loss_plots(loss_metrics)

    # Create performance plots
    if perf_metrics:
        _create_perf_plots(perf_metrics)


def _create_loss_plots(loss_metrics: dict) -> None:
    """Create the loss plots."""
    if not any(not df.empty for df in loss_metrics.values()):
        return
    st.markdown("### Training Loss")
    loss_cols = st.columns(2)

    for i, (metric_name, metric_df) in enumerate(loss_metrics.items()):
        col = loss_cols[i % 2]
        with col:
            st.markdown(f"**{metric_name.replace('losses/', '')}**")
            if len(metric_df) > 1:
                st.line_chart(
                    metric_df, x="step", y="value", use_container_width=True
                )
            elif len(metric_df) == 1:
                st.metric("Value", f"{metric_df.iloc[0]['value']:.4f}")


def _create_perf_plots(perf_metrics: dict) -> None:
    """Create the performance plots."""
    if not any(not df.empty for df in perf_metrics.values()):
        return
    st.markdown("### Performance Metrics")
    perf_cols = st.columns(3)

    # Steps per second
    if (
        "perf_stats/steps_per_sec" in perf_metrics
        and not perf_metrics["perf_stats/steps_per_sec"].empty
    ):
        perf_cols[0].markdown("**Training Speed (Steps/Second)**")
        perf_cols[0].line_chart(
            perf_metrics["perf_stats/steps_per_sec"],
            x="step",
            y="value",
            use_container_width=True,
        )

    # Training time
    if (
        "perf_stats/total_training_time_hours" in perf_metrics
        and not perf_metrics["perf_stats/total_training_time_hours"].empty
    ):
        perf_cols[1].markdown("**Total Training Time (Hours)**")
        perf_cols[1].line_chart(
            perf_metrics["perf_stats/total_training_time_hours"],
            x="step",
            y="value",
            use_container_width=True,
        )

    # Data throughput
    if (
        "perf_stats/data_points_per_sec_global" in perf_metrics
        and not perf_metrics["perf_stats/data_points_per_sec_global"].empty
    ):
        perf_cols[2].markdown("**Data Throughput (Points/Second)**")
        perf_cols[2].line_chart(
            perf_metrics["perf_stats/data_points_per_sec_global"],
            x="step",
            y="value",
            use_container_width=True,
        )
