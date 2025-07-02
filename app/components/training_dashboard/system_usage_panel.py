import pandas as pd
import psutil
import pynvml
import streamlit as st

@st.fragment(run_every=1)
def display_system_usage_panel():
    """Display the system resource usage panel."""
    st.subheader("System Resource Usage")
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_util, gpu_mem, gpu_temp = [], [], []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.append(util_info.gpu)
            gpu_mem.append(mem_info.used / (1024**3))
            gpu_temp.append(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )

        cpu_util_val = psutil.cpu_percent()

        if "history" not in st.session_state or not st.session_state.history:
            st.session_state.history = {
                "gpu_util": [],
                "gpu_mem": [],
                "gpu_temp": [],
                "cpu_util": [],
            }

        history = st.session_state.history
        history["gpu_util"].append(
            sum(gpu_util) / len(gpu_util) if gpu_util else 0
        )
        history["gpu_mem"].append(sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0)
        history["gpu_temp"].append(
            sum(gpu_temp) / len(gpu_temp) if gpu_temp else 0
        )
        history["cpu_util"].append(cpu_util_val)

        for key in history:
            history[key] = history[key][-100:]

        if len(history["cpu_util"]) > 1:
            chart_cols = st.columns(4)
            chart_cols[0].line_chart(
                pd.DataFrame(history["gpu_util"], columns=["Usage (%)"]),
                use_container_width=True,
            )
            chart_cols[0].caption("GPU Utilization (%)")
            chart_cols[1].line_chart(
                pd.DataFrame(history["gpu_mem"], columns=["Memory (GB)"]),
                use_container_width=True,
            )
            chart_cols[1].caption("GPU Memory Usage (GB)")
            chart_cols[2].line_chart(
                pd.DataFrame(history["gpu_temp"], columns=["Temp (°C)"]),
                use_container_width=True,
            )
            chart_cols[2].caption("GPU Temperature (°C)")
            chart_cols[3].line_chart(
                pd.DataFrame(history["cpu_util"], columns=["Usage (%)"]),
                use_container_width=True,
            )
            chart_cols[3].caption("CPU Utilization (%)")
        else:
            st.info("Collecting system usage data...")

    except pynvml.NVMLError:
        st.warning("NVIDIA GPU not detected.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
