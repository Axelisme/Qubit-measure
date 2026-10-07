"""Keep notebook behavior tests independent of the host software environment."""

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.context import RunContext
from zcu_tools.notebook import adapter


@pytest.fixture(autouse=True)
def software_provenance(monkeypatch: pytest.MonkeyPatch) -> SoftwareProvenance:
    """Replace external capture, not notebook acquisition or evidence lifecycle."""
    provenance = SoftwareProvenance(
        software_versions={"python": "test", "zcu-tools": "test", "numpy": "test"},
        git_commit=None,
        git_dirty=None,
        qick_version=None,
        soc_fingerprint=None,
        hostname="test-host",
    )

    def capture(*, context: RunContext | None = None) -> SoftwareProvenance:
        return provenance

    monkeypatch.setattr(adapter, "capture_software_provenance", capture)
    return provenance
