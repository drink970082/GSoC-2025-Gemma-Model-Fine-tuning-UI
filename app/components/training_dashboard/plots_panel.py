import pandas as pd
import streamlit as st

from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_plots_panel(training_service: TrainingService):
    """Display the core performance plots panel."""
    st.subheader("Core Performance Plots")

    loss_metrics = training_service.get_loss_metrics()
    perf_metrics = training_service.get_performance_metrics()

    if not loss_metrics and not perf_metrics:
        st.info("Waiting for first metric to be logged...")
        return

    # Create loss plots
    if loss_metrics:
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
                else:
                    st.metric("Value", f"{metric_df.iloc[0]['value']:.4f}")

    # Create performance plots
    if perf_metrics:
        st.markdown("### Performance Metrics")

        # Steps per second
        if "perf_stats/steps_per_sec" in perf_metrics:
            st.markdown("**Training Speed (Steps/Second)**")
            st.line_chart(
                perf_metrics["perf_stats/steps_per_sec"],
                x="step",
                y="value",
                use_container_width=True,
            )

        # Training time
        if "perf_stats/total_training_time_hours" in perf_metrics:
            st.markdown("**Total Training Time (Hours)**")
            st.line_chart(
                perf_metrics["perf_stats/total_training_time_hours"],
                x="step",
                y="value",
                use_container_width=True,
            )

        # Data throughput
        if "perf_stats/data_points_per_sec_global" in perf_metrics:
            st.markdown("**Data Throughput (Points/Second)**")
            st.line_chart(
                perf_metrics["perf_stats/data_points_per_sec_global"],
                x="step",
                y="value",
                use_container_width=True,
            )
