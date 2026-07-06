"""Lightweight smoke tests.

These intentionally avoid importing the Streamlit app or heavy ML deps so CI
stays fast and only needs the core requirements. They verify that:
  1. Every module under src/ compiles (syntax is valid).
  2. project_paths resolves the repository root and version layout.
  3. Core v1 cleaned datasets are present and non-empty.
"""
from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _python_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


@pytest.mark.parametrize("py_file", _python_files(), ids=lambda p: p.name)
def test_source_compiles(py_file: Path) -> None:
    """Each source file must be syntactically valid Python."""
    py_compile.compile(str(py_file), doraise=True)


def test_project_paths_importable() -> None:
    """project_paths should expose the repo root and a version identifier."""
    import sys

    sys.path.insert(0, str(SRC))
    import project_paths  # noqa: E402

    assert project_paths.REPO_ROOT.exists()
    assert isinstance(project_paths.VERSION, str)


def test_v1_clean_data_present() -> None:
    """Key v1 cleaned outputs should exist so the default dashboard can load."""
    clean = REPO_ROOT / "v1" / "data_clean"
    expected = ["merged_ml_country.csv", "who_2021_clean.csv"]
    for name in expected:
        path = clean / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"
