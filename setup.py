from setuptools import setup, find_packages

setup(
    name="mask-learning-tool",
    version="0.1.0",
    packages=find_packages(exclude=("examples", "scripts", "tests", "data")),
    install_requires=[
        "torch>=2.1",
        "numpy>=1.23",
        "scikit-learn>=1.1",
        "matplotlib>=3.7",
        "PyYAML>=6.0",
        "tqdm>=4.65",
    ],
    description="Lightweight training & evaluation tool for mask/regression models (DL).",
    author="You",
    python_requires=">=3.9",
)
