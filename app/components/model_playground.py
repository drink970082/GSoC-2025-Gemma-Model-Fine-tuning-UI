import plotly.graph_objects as go
import streamlit as st


def show_model_playground(model):
    """Display the model playground with training statistics and inference interface."""


    # Show training metrics
    st.subheader("Training Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Loss", f"{model['metrics']['final_loss']:.3f}")
    with col2:
        st.metric("Training Time", model["metrics"]["training_time"])
    with col3:
        st.metric("GPU Memory Used", model["metrics"]["gpu_memory"])

    # Show training plot
    st.subheader("Training Metrics")
    fig = go.Figure(model["training_plot"])
    st.plotly_chart(fig, use_container_width=True)

    # Inference playground
    st.subheader("Test Your Model")
    test_input = st.text_area("Enter a prompt to test")
    if st.button("Generate"):
        st.write("Generated response will appear here...")
