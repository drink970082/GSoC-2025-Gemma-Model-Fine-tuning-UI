from setuptools import find_packages, setup

setup(
    name="gemma-fine-tuning-ui",
    version="0.1.0",
    description="A UI for fine-tuning Gemma models",
    author="Chen-Hao Wu",
    author_email="howdywu@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "plotly",
        "requests",
        "jax",
        "jax[cuda12]",
        "gemma",
        "ipykernel",
    ],
    python_requires=">=3.8",
)
