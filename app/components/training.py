import time

import plotly.graph_objects as go
import streamlit as st


def show_training():
    """Display the training interface with progress and metrics."""

    # Get training configuration
    config = st.session_state.training_config

    # Training progress
    st.subheader("Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Training loop
    if st.session_state.training_active:
        # Abort button
        if st.button("Abort Training", type="secondary"):
            st.session_state.training_active = False
            st.warning("Training aborted!")
            st.session_state.page = "home"
            st.rerun()

        # Clear previous training data
        st.session_state.training_data = {
            "loss": [],
            "learning_rate": [],
            "gpu_memory": [],
            "epochs": [],
        }

        # Simulate training progress
        for i in range(100):
            if not st.session_state.training_active:
                break

            # Update progress
            progress_bar.progress(i + 1)

            # Calculate metrics
            epoch = i // 20 + 1
            loss = 1.0 - i / 100
            learning_rate = 0.001 * (1 - i / 100)
            gpu_memory = 12.5 * (1 - i / 100)

            # Update status
            status_text.text(
                f"Epoch {epoch}/{config['epochs']} - Loss: {loss:.3f}"
            )

            # Store metrics
            st.session_state.training_data["loss"].append(loss)
            st.session_state.training_data["learning_rate"].append(
                learning_rate
            )
            st.session_state.training_data["gpu_memory"].append(gpu_memory)
            st.session_state.training_data["epochs"].append(epoch)

            # Update plot
            fig = go.Figure()

            # Add loss trace
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.training_data["loss"],
                    name="Loss",
                    line=dict(color="red"),
                )
            )

            # Add learning rate trace
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.training_data["learning_rate"],
                    name="Learning Rate",
                    line=dict(color="blue"),
                )
            )

            # Add epoch markers
            for e in range(1, epoch + 1):
                epoch_start = (e - 1) * 20
                if epoch_start < len(st.session_state.training_data["loss"]):
                    fig.add_vline(
                        x=epoch_start,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Epoch {e}",
                        annotation_position="top right",
                    )

            fig.update_layout(
                title="Training Metrics",
                xaxis_title="Steps",
                yaxis_title="Value",
                height=400,
                showlegend=True,
            )
            plot_placeholder.plotly_chart(
                fig, use_container_width=True, key=f"training_progress_plot_{i}"
            )

            # Show current metrics in columns
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Loss", f"{loss:.3f}")
                with col2:
                    st.metric("Learning Rate", f"{learning_rate:.6f}")
                with col3:
                    st.metric("GPU Memory", f"{gpu_memory:.1f}GB")

            # Simulate training time
            time.sleep(0.1)

        # Store model info and reset training state
        model_info = {
            "name": config["name"],
            "status": "success",
            "date": time.strftime("%Y-%m-%d"),
            "metrics": {
                "final_loss": loss,
                "training_time": "2h 30m",
                "gpu_memory": f"{gpu_memory:.1f}GB",
            },
            "training_data": {
                "loss": st.session_state.training_data["loss"],
                "learning_rate": st.session_state.training_data[
                    "learning_rate"
                ],
                "gpu_memory": st.session_state.training_data["gpu_memory"],
                "epochs": st.session_state.training_data["epochs"],
            },
            "training_plot": fig.to_dict(),
        }

        # Show completion screen
        st.success("Training completed successfully!")
        st.session_state.current_model = model_info
        st.session_state.training_active = False
        st.rerun()
