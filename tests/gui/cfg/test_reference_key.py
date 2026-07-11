from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    is_custom_reference_key,
    make_custom_reference_key,
    parse_custom_reference_key,
)


def test_custom_reference_key_round_trip() -> None:
    key = make_custom_reference_key("Pulse Readout")
    assert key == "<Custom:Pulse Readout>"
    assert parse_custom_reference_key(key) == "Pulse Readout"
    assert is_custom_reference_key(key) is True
    assert parse_custom_reference_key("library-entry") is None
    assert is_custom_reference_key("library-entry") is False


@pytest.mark.parametrize(
    "key", ("<Custom:", "<Custom:>", "<Custom:Pulse", "<Custom:Pulse>>tail")
)
def test_malformed_custom_reference_key_fails_fast(key: str) -> None:
    with pytest.raises(ValueError, match="Invalid custom reference key"):
        parse_custom_reference_key(key)


@pytest.mark.parametrize("label", ("", None))
def test_make_custom_reference_key_requires_non_empty_string(label: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        make_custom_reference_key(label)  # type: ignore[arg-type]


def test_make_custom_reference_key_rejects_reserved_delimiter() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        make_custom_reference_key("A>B")
