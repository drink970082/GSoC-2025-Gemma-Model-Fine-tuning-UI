import glob
import os

import pandas as pd
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from config.training_config import TENSORBOARD_LOGDIR


@st.fragment(run_every=1)
def load_event_data(log_dir: str) -> dict[str, pd.DataFrame]:
    """Load event data from TensorBoard logs.

    Recursively searches for event files in the log directory and its subdirectories.
    Returns a dictionary mapping metric names to DataFrames containing the metric values.
    """
    data = {}
    try:
        # Find event files
        event_files = []
        for root, _, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, file))

        if not event_files:
            return data

        # Use the latest event file
        latest_event_file = max(event_files, key=os.path.getmtime)

        # Load event data
        event_acc = EventAccumulator(
            latest_event_file, size_guidance={"scalars": 0, "tensors": 0}
        )
        event_acc.Reload()

        # Get available tags
        tags = event_acc.Tags()

        # Process scalar metrics
        for tag in tags.get("scalars", []):
            events = event_acc.Scalars(tag)
            data[tag] = pd.DataFrame(
                [(e.wall_time, e.step, e.value) for e in events],
                columns=["wall_time", "step", "value"],
            )

        # Process tensor metrics
        for tag in tags.get("tensors", []):
            if "loss" in tag.lower() or "perf" in tag.lower():
                try:
                    events = event_acc.Tensors(tag)
                    data[tag] = pd.DataFrame(
                        [
                            (
                                e.wall_time,
                                e.step,
                                float(e.tensor_proto.float_val[0]),
                            )
                            for e in events
                        ],
                        columns=["wall_time", "step", "value"],
                    )
                except Exception:
                    continue

    except Exception:
        pass

    return data


def display_tensorboard_iframe(port: int = 6007) -> None:
    """Display TensorBoard directly in the Streamlit app using an iframe.

    Args:
        port: The port number where TensorBoard is running (default: 6006)
    """
    st.markdown(
        f"""
        <iframe
            src="http://localhost:{port}"
            width="100%"
            height="800"
            frameborder="0"
            allowfullscreen
        ></iframe>
        """,
        unsafe_allow_html=True,
    )
