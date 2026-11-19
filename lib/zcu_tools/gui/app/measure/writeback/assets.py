"""Measure's preview and selected-Apply policy for arb asset proposals."""

from zcu_tools.cfg_model import ModuleSpec, module_arb_keys
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.types import DocumentSetCapture
from zcu_tools.resources.context.library import LibrarySnapshot
from zcu_tools.resources.entry import writeback as concrete
from zcu_tools.resources.waveform_assets import (
    ArbAssetUpdateStatus,
    arb_sample_summary,
)

from .contracts import ArbAssetSummary, ArbAssetValues
from .items import ArbAssetWriteback


def concrete_arb_asset(item: ArbAssetWriteback) -> concrete.ArbAssetWriteback:
    """Lower the proposed asset key and samples to an entry writeback item."""
    return concrete.ArbAssetWriteback(item.target_name, item.samples)


def arb_asset_current(
    item: ArbAssetWriteback, current: DocumentSetCapture
) -> ArbAssetUpdateStatus:
    """Compare samples in the current entry without modifying its asset files."""
    return concrete.arb_asset_update_status(
        current.capture_storage().database_path, concrete_arb_asset(item)
    )


def preview_arb_asset(
    item: ArbAssetWriteback, path: str, current: DocumentSetCapture | None
) -> ArbAssetValues:
    """Return preview facts for path, using None for status without an open entry."""
    status = None if current is None else arb_asset_current(item, current)
    samples = item.samples
    return ArbAssetValues(
        actual_path=path,
        current=status,
        proposed=ArbAssetSummary(
            **arb_sample_summary(samples), peak_abs=samples.peak_abs
        ),
        unit=None,
        warning=(
            f"Apply will fail: arb asset {item.target_name!r} has different samples."
            if status == "different"
            else None
        ),
    )


def require_arb_references(
    spec: ModuleSpec,
    library: LibrarySnapshot,
    selected_assets: set[str],
    *,
    session_id: str,
    target: str,
) -> None:
    """Reject reachable arb keys missing from the active entry and selected batch.

    Use the active library's typed child Specs and bound asset owner. Do not
    read samples, check unrelated library references or write anything.
    InvalidInputError names the proposal ID, target and missing key.
    """
    assets = library.validation_context().assets
    for key in sorted(module_arb_keys(spec, library) - selected_assets):
        if assets is not None and assets.exists(key):
            continue
        raise InvalidInputError(
            f"Module proposal {session_id!r} ({target}) requires arb asset {key!r}; "
            "publish it in the same draft or select an existing asset"
        )
