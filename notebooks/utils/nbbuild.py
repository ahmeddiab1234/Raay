"""Minimal nbformat writer shared by the ``_build_eda*.py`` generators.

The EDA notebooks are generated programmatically rather than hand-edited, so
they were re-implementing the same six helpers inline. Hand-rolling a notebook
is easy to get subtly wrong (``source`` must be a *list of lines*, not one
string; nbformat 4.5 requires a per-cell ``id``; ``indent=1`` is what Jupyter
writes), and the duplication meant a fix had to be applied twice.

Deliberately dependency-free: these scripts run under bare ``python3`` in a
notebook environment, not the project venv.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

NBFORMAT = 4
NBFORMAT_MINOR = 5


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _lines(text: str) -> list[str]:
    """Split text into lines with trailing newlines (except the last)."""
    raw = text.split("\n")
    return [line + "\n" for line in raw[:-1]] + [raw[-1]] if raw else []


def md(source: str) -> dict[str, Any]:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
        "id": _uid(),
    }


def code(source: str) -> dict[str, Any]:
    """An unexecuted code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _lines(source.strip()),
        "execution_count": None,
        "outputs": [],
        "id": _uid(),
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap assembled cells in an nbformat 4.5 document."""
    return {
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }


def write(cells: list[dict[str, Any]], out: str | Path) -> Path:
    """Write the assembled notebook to ``out`` and return the path."""
    path = Path(out)
    path.write_text(
        json.dumps(notebook(cells), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"✅ Wrote {path}")
    return path
