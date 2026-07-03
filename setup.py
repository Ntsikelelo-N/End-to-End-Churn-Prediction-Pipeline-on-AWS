"""
setup.py — Makes churn_pipeline importable as a local package.

Running `pip install -e .` from the repo root means every notebook, script,
and test can do `from churn_pipeline import ...` without path hacks.
"""

from setuptools import find_packages, setup

setup(
    name="churn-pipeline",
    version="0.1.0",
    description="End-to-end customer churn prediction pipeline on AWS",
    author="Ntsikelelo Jantjie",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "boto3>=1.34",
        "requests>=2.31",
    ],
    extras_require={
        "ml": ["xgboost>=2.0"],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "black>=24.0",
            "flake8>=7.0",
        ],
    },
)
