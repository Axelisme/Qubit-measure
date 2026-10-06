from pathlib import Path

import pytest


@pytest.fixture
def document_path(tmp_path: Path) -> Path:
    path = tmp_path / "document.yaml"
    path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n  left: 1.0\n  right: 2.0\n",
        encoding="utf-8",
    )
    return path
