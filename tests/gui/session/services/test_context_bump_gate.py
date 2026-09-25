"""ContextService md/ml writers retain the context resource-version bump."""

from __future__ import annotations

from pathlib import Path

import zcu_tools.gui.session.services as session_services_pkg

from tests.gui._context_bump_gate import missing_context_bumps


def test_md_ml_writers_bump_context_version() -> None:
    assert session_services_pkg.__file__ is not None
    path = Path(session_services_pkg.__file__).parent / "context.py"
    offenders = missing_context_bumps(path)
    assert not offenders, (
        "every function that writes MetaDict/ModuleLibrary content must "
        'self.version.bump("context") (hidden-contract gate). '
        f"Missing the bump: {offenders}"
    )
