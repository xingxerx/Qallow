#!/usr/bin/env python3
"""
Setup script for ALG - Quantum Algorithm Optimizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qallow-alg",
    version="1.0.0",
    author="Qallow Team",
    author_email="dev@qallow.io",
    description="Quantum Algorithm Optimizer for Qallow (QAOA + SPSA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xingxerx/Qallow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qiskit>=0.39.0",
        "qiskit-aer>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "alg=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

