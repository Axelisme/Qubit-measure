"""Shared autofluxdep artifact path helper tests."""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.services.artifact_paths import safe_artifact_slug


def test_safe_artifact_slug_preserves_fallback_semantics() -> None:
    assert safe_artifact_slug("   ", fallback="node") == "node"
    assert safe_artifact_slug("   ") == "unnamed"
    assert safe_artifact_slug("a/b c") == "a-b-c"
