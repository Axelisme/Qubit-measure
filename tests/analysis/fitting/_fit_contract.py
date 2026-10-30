"""Assertions shared by the public named fitting-result contracts."""

from collections.abc import Sequence

import pytest


def assert_frozen_fields(result: object, names: Sequence[str]) -> None:
    """Check assignment and deletion fail for each named result attribute."""
    for name in names:
        with pytest.raises(AttributeError):
            setattr(result, name, None)
        with pytest.raises(AttributeError):
            delattr(result, name)
