from setuptools import setup, find_packages

setup(
    name="datagen",
    version="0.1.0",
    description="Synthetic data generation library for LLM training",
    author="DataGen Team",
    author_email="info@datagen.ai",  # Example email
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "transformers>=4.18.0",
        "openai>=0.27.0",
        "datasets>=2.0.0",
        "click>=8.0.0",  # For CLI
        "pyyaml>=6.0",   # For configuration
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "datagen=datagen.cli:main",
        ],
    },
) 