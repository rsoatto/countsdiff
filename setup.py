#!/usr/bin/env python3
"""
Setup script for SNP Blackout Diffusion package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "CountsDiff: Diffusion on the natural numbers for imputation and generation of count-based data"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.0',
        'diffusers>=0.21.0',
        'transformers>=4.21.0',
        'accelerate>=0.20.0',
    ]

setup(
    name="counstdiff",
    version="0.1.0",
    author="Anonymous",
    author_email="Anonymous",
    description="CountsDiff: Diffusion on the natural numbers for imputation and generation of count-based data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/counstdiff",
    packages=find_packages(where="src") + find_packages(include=("baselines", "baselines.*", "scripts", "scripts.*"))
,
    package_dir={"": "src",
                 "baselines": "baselines",
                 "scripts": "scripts"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
        "neptune": [
            "neptune-client>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snp-train=countsdiff.cli:train_cli",
            "snp-generate=countsdiff.cli:generate_cli", 
            "countsdiff=countsdiff.cli:main",
            "snp-preprocess=countsdiff.data.preprocessing:preprocess_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "countsdiff": ["configs/*.yaml"],
    },
    zip_safe=False,
)
