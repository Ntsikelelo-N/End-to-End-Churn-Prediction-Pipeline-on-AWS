"""
tests/conftest.py — Shared pytest configuration.

Adds src/ to sys.path so tests can import churn_pipeline without requiring
`pip install -e .` to have been run first. pytest loads this file
automatically before collecting any test module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
