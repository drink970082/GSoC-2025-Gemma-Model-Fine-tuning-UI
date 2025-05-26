import streamlit as st
import pandas as pd
import uuid
import time
import numpy as np

st.set_page_config(page_title="Gemma Fine-tuning Pipeline", layout="wide")
st.title("Gemma Model Fine-tuning UI")

# Step Navigation
step = st.sidebar.radio("Pipeline Steps", ["1. Data Exploration", "2. Hyperparameter Tuning & Training", "3. Summary & Export"])

# Shared Session State
if "data" not in st.session_state:
    st.session_state.data = None
if "input_columns" not in st.session_state:
    st.session_state.input_columns = []
if "output_columns" not in st.session_state:
    st.session_state.output_columns = []
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 5e-5
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 8
if "epochs" not in st.session_state:
    st.session_state.epochs = 3
if "training_log" not in st.session_state:
    st.session_state.training_log = {}
if "model_id" not in st.session_state:
    st.session_state.model_id = f"tunedModels/gemma-model-{uuid.uuid4().hex[:8]}"
if "training_time" not in st.session_state:
    st.session_state.training_time = "N/A"

# Step 1: Data Exploration
if step == "1. Data Exploration":
    st.header("Upload & Explore Dataset")
    file = st.file_uploader("Upload a CSV, JSONL, or TXT file", type=["csv", "jsonl", "txt"])

    if file:
        file_type = file.name.split(".")[-1]
        if file_type == "csv":
            st.session_state.data = pd.read_csv(file)
        elif file_type == "jsonl":
            st.session_state.data = pd.read_json(file, lines=True)
        elif file_type == "txt":
            st.session_state.data = pd.DataFrame({"text": file.read().decode("utf-8").splitlines()})

        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head())

        input_columns = st.multiselect("Select the input column(s)", options=st.session_state.data.columns)
        output_columns = st.multiselect("Select the output column(s)", options=st.session_state.data.columns)
        st.subheader("Preprocessing Options")
        lowercase = st.checkbox("Convert text to lowercase")
        strip_whitespace = st.checkbox("Strip whitespace")

        if lowercase:
            for col in input_columns:
                st.session_state.data[col] = st.session_state.data[col].astype(str).str.lower()
        if strip_whitespace:
            for col in input_columns:
                st.session_state.data[col] = st.session_state.data[col].astype(str).str.strip()

        st.subheader("Data Summary")
        st.write("Total Samples:", len(st.session_state.data))
        if input_columns:
            input_text = " ".join([str(st.session_state.data[col].iloc[0]) for col in input_columns])
            st.write("Input Example:", input_text)
        if output_columns:
            output_text = " ".join([str(st.session_state.data[col].iloc[0]) for col in output_columns])
            st.write("Output Example:", output_text)

# Step 2: Hyperparameter Tuning and Training
elif step == "2. Hyperparameter Tuning & Training":
    st.header("Configure Hyperparameters & Train")
    if st.session_state.data is None:
        st.warning("Please upload and explore your dataset in Step 1.")
    else:
        st.subheader("Hyperparameter Configuration")
        st.session_state.learning_rate = st.number_input("Learning Rate", 1e-6, 1.0, st.session_state.learning_rate, format="%e", help="Controls how much to update the model weights.")
        st.session_state.batch_size = st.slider("Batch Size", 1, 64, st.session_state.batch_size, help="Number of samples per training step.")
        st.session_state.epochs = st.slider("Epochs", 1, 20, st.session_state.epochs, help="Number of passes through the entire dataset.")

        st.subheader("Training Progress")
        if st.button("Start Training"):
            st.info("Training started... (placeholder)")
            start = time.time()

            loss = np.cumsum(np.random.randn(st.session_state.epochs) * -2 + 10).clip(min=0.1).tolist()
            acc = np.cumsum(np.random.randn(st.session_state.epochs) * 1.5 + 60).clip(min=50, max=100).tolist()

            st.session_state.training_log = {
                "loss": loss,
                "acc": acc,
                "samples": [
                    {"input": " ".join([str(st.session_state.data[col].iloc[0]) for col in st.session_state.input_columns]), "output": f"Generated response after epoch {i+1}"}
                    for i in range(st.session_state.epochs)
                ],
            }

            st.session_state.training_time = f"{int(time.time() - start)}s"
            st.success("Training complete!")

            st.line_chart({"Loss": loss, "Accuracy": acc}, height=200)
            for i, sample in enumerate(st.session_state.training_log["samples"]):
                st.markdown(f"**Epoch {i+1} Sample Output**")
                st.write("Input:", sample["input"])
                st.write("Generated Output:", sample["output"])

# Step 3: Summary & Export
elif step == "3. Summary & Export":
    st.header("Summary & Model Export")
    st.write("Here you can download your fine-tuned model.")
    st.download_button("Download PyTorch Model", data=b"placeholder for .bin", file_name="gemma_finetuned_model.bin")

    st.subheader("Training Summary")
    st.markdown(
        """
    #### Tuning details
    - **Model ID:** `{}`  
    - **Base model:** Gemma (placeholder)  
    - **Total training time:** {}  
    - **Tuned examples:** {} examples  
    - **Epochs:** {}  
    - **Batch size:** {}  
    - **Learning rate:** {}  
    """.format(
            st.session_state.model_id,
            st.session_state.training_time,
            len(st.session_state.data) if st.session_state.data is not None else "N/A",
            st.session_state.epochs,
            st.session_state.batch_size,
            st.session_state.learning_rate,
        )
    )

    if st.session_state.training_log:
        st.line_chart({"Loss": st.session_state.training_log["loss"], "Accuracy": st.session_state.training_log["acc"]})

st.markdown("---")
st.caption("This is a demo page showing minimal functionality. The actual implementation may vary.")
