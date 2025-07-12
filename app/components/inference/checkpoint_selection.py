import streamlit as st
from pathlib import Path
from config.app_config import get_config
import os
import shutil
from typing import Optional
from backend.core.model import Model
from gemma import gm
import json
from config.dataclass import ModelConfig, LoraParams, DpoParams


config = get_config()


def get_checkpoint_path(work_dir: str) -> str:
    checkpoint_path = os.path.join(work_dir, "checkpoints")
    subdirs = [p for p in Path(checkpoint_path).iterdir() if p.is_dir()]
    return subdirs[0]


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


def parse_model_config(model_config_dict: dict) -> ModelConfig:
    # Handle the parameters field
    parameters = None
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
    params = Model.load_trained_params(get_checkpoint_path(checkpoint_path))
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


def show_checkpoint_selection():
    st.subheader("Checkpoint Management")
    checkpoints = list_checkpoints(config.CHECKPOINT_FOLDER)
    if not checkpoints:
        st.warning("No checkpoints found.")
        return
    selected_checkpoint = st.selectbox(
        "Select a checkpoint for inference:", checkpoints, index=0
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(
            "Load checkpoint", type="primary", use_container_width=True
        ):
            with st.spinner(f"Loading checkpoint: {selected_checkpoint}"):
                checkpoint_path = str(
                    Path(config.CHECKPOINT_FOLDER) / selected_checkpoint
                )
                if not os.path.exists(checkpoint_path):
                    st.error(f"Checkpoint {selected_checkpoint} not found.")
                    return
                sampler, tokenizer = load_model(checkpoint_path)
                if not sampler:
                    st.error(
                        f"Failed to load checkpoint: {selected_checkpoint}"
                    )
                    return
                st.session_state.sampler = sampler
                st.session_state.tokenizer = tokenizer
            st.success(f"Loaded checkpoint: {selected_checkpoint}")

    with col2:
        if st.button("Delete", type="secondary", use_container_width=True):
            if delete_checkpoint(config.CHECKPOINT_FOLDER, selected_checkpoint):
                st.success(f"Deleted checkpoint: {selected_checkpoint}")
                st.session_state.sampler = None
                st.session_state.tokenizer = None
                st.rerun()
            else:
                st.error("Failed to delete checkpoint.")
