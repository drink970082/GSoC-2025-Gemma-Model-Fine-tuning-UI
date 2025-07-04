import streamlit as st
from pathlib import Path
import os
import shutil
from config.app_config import get_config
from backend.core.model import Model
from gemma import gm
from typing import Optional
import json
from config.dataclass import ModelConfig, LoraParams, DpoParams

config = get_config()


def parse_model_config(model_config_dict: dict) -> ModelConfig:
    # Handle the parameters field
    parameters = None
    model_config_dict = config["model_config"]
    if model_config_dict.get("parameters"):
        if model_config_dict["method"] == "LoRA":
            parameters = LoraParams(**model_config_dict["parameters"])
        elif model_config_dict["method"] == "DPO":
            parameters = DpoParams(**model_config_dict["parameters"])
    return ModelConfig(
        model_variant=model_config_dict["model_variant"],
        epochs=model_config_dict["epochs"],
        learning_rate=model_config_dict["learning_rate"],
        method=model_config_dict["method"],
        parameters=parameters,
    )


def list_checkpoints(work_dir: str) -> list[str]:
    """Return a list of checkpoint directory names, sorted by creation time (newest first)."""
    if not os.path.exists(work_dir):
        return []
    subdirs = [p for p in Path(work_dir).iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    return [p.name for p in subdirs]


def get_latest_checkpoint(work_dir: str) -> Optional[str]:
    """Return the path of the most recently created checkpoint directory."""
    checkpoints = list_checkpoints(work_dir)
    if not checkpoints:
        return None
    return checkpoints[0]


def delete_checkpoint(work_dir: str, checkpoint_name: str) -> bool:
    """Delete the specified checkpoint directory."""
    checkpoint_path = Path(work_dir) / checkpoint_name
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        shutil.rmtree(checkpoint_path)
        return True
    return False


def load_model(
    checkpoint_path: Optional[str] = None,
) -> Optional[tuple[gm.text.ChatSampler, gm.text.Gemma3Tokenizer]]:
    """Load the most recent trained model if available."""
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
        if not checkpoint_path:
            return None, None
    else:
        if not os.path.exists(checkpoint_path):
            return None, None

    # Load trained parameters from the latest checkpoint
    params = Model.load_trained_params(checkpoint_path)
    model_config: ModelConfig = parse_model_config(
        json.load(open(f"{checkpoint_path}/model_config.json"))
    )

    if model_config.method == "LoRA":
        model = Model.create_lora_model(
            model_config.model_variant,
            model_config.parameters.lora_rank,
        )
    else:
        model = Model.create_standard_model(model_config.model_variant)

    # Create tokenizer
    tokenizer = gm.text.Gemma3Tokenizer()

    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )
    return sampler, tokenizer


def show_inference_playground():
    """
    Displays the interactive inference playground components, including
    the prompt input, generate button, and response display.
    """
    # Checkpoint selection and management
    if "sampler" not in st.session_state:
        st.session_state.sampler = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None

    st.subheader("Checkpoint Management")
    checkpoints = list_checkpoints(config.CHECKPOINT_FOLDER)
    if not checkpoints:
        st.warning("No checkpoints found.")
        return
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_checkpoint = st.selectbox(
            "Select a checkpoint for inference:", checkpoints, index=0
        )
    with col2:
        if st.button("Delete", type="secondary"):
            if delete_checkpoint(config.CHECKPOINT_FOLDER, selected_checkpoint):
                st.success(f"Deleted checkpoint: {selected_checkpoint}")
                st.rerun()
            else:
                st.error("Failed to delete checkpoint.")

    if st.button("Load checkpoint", type="primary"):
        with st.spinner(f"Loading checkpoint: {selected_checkpoint}"):
            checkpoint_path = str(
                Path(config.CHECKPOINT_FOLDER) / selected_checkpoint
            )
            sampler, tokenizer = load_model(checkpoint_path)
            if not sampler:
                st.error(f"Failed to load checkpoint: {selected_checkpoint}")
                return
            st.session_state.sampler = sampler
            st.session_state.tokenizer = tokenizer
        st.success(f"Loaded checkpoint: {selected_checkpoint}")

    st.divider()
    # Inference interface
    st.subheader("Inference Playground")
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Type your message here...",
        height=100,
    )
    if st.button("Generate Response", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating response..."):
            try:
                if "sampler" not in st.session_state:
                    st.error(
                        "No sampler found. Please load a checkpoint first."
                    )
                    return
                response = st.session_state.sampler.chat(prompt)

                st.subheader("Response")
                st.write(response)

                # Show token info if the tokenizer is available
                if st.session_state.tokenizer:
                    input_tokens = len(
                        st.session_state.tokenizer.encode(prompt)
                    )
                    output_tokens = len(
                        st.session_state.tokenizer.encode(response)
                    )
                    st.caption(
                        f"Input tokens: {input_tokens} | Output tokens: {output_tokens}"
                    )

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
