from setuptools import find_packages, setup

setup(
    name="gemma_fine_tuning_ui",
    version="0.1.0",
    description="A UI for fine-tuning Gemma models",
    author="Chen-Hao Wu",
    author_email="howdywu@gmail.com",
    packages=find_packages(include=['app', 'app.*', 'services', 'services.*', 'config', 'config.*', 'backend', 'backend.*']),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "requests",
        "gemma",
        "datasets",
        "pynvml",
    ],
    python_requires=">=3.8",
)
