from setuptools import setup, find_packages

setup(
    name="driftwatch",
    version="0.1.0",
    description="Python SDK for the DriftWatch ML monitoring platform",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.0.0",
    ],
)
