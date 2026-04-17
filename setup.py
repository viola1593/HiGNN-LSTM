from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="twsa-hignn-lstm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Global Water Storage Prediction with HiGNN-LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TWSA_HiGNN-LSTM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "xarray>=0.21.0",
        "netCDF4>=1.6.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "pytorch-lightning>=1.7.0",
        "torchmetrics>=0.10.0",
        "torch-geometric>=2.0.0",
        "torch-scatter>=2.0.0",
        "torch-sparse>=0.6.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4",
        "pydantic>=1.8",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "pytest>=6.0.0",
        ],
        "analysis": [
            "shap>=0.41.0",
            "captum>=0.7.0",
            "plotly>=5.0.0",
        ],
    },
)
