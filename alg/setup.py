# [REVIEWED] # [REVIEWED] # [REVIEWED] #!/usr/bin/env python3
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] Setup script for ALG - Quantum Algorithm Optimizer
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] with open("README.md", "r", encoding="utf-8") as f:
# [REVIEWED] # [REVIEWED] # [REVIEWED]     long_description = f.read()
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] setup(
# [REVIEWED] # [REVIEWED] # [REVIEWED]     name="qallow-alg",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     version="1.0.0",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     author="Qallow Team",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     author_email="dev@qallow.io",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     description="Quantum Algorithm Optimizer for Qallow (QAOA + SPSA)",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     long_description=long_description,
# [REVIEWED] # [REVIEWED] # [REVIEWED]     long_description_content_type="text/markdown",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     url="https://github.com/xingxerx/Qallow",
# [REVIEWED] # [REVIEWED] # [REVIEWED]     packages=find_packages(),
# [REVIEWED] # [REVIEWED] # [REVIEWED]     classifiers=[
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
    ],
    entry_points={
        "console_scripts": [
            "alg=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

